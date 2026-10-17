"""Supervised periodic scalar regression with MSE and a QFT-layout QNN."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qftlab.circuits.qft import PARAMETER_NAMES, QFT_PARAMETERS, build_qft_ansatz, reference_unitary
from qftlab.objectives import process_fidelity
from qiskit.quantum_info import Operator


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    import tomllib
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _target(x: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    result = np.full_like(np.asarray(x, dtype=float), float(data["target_offset"]))
    for k, cosine, sine in _target_components(data):
        result += cosine * np.cos(2 * np.pi * k * x) + sine * np.sin(2 * np.pi * k * x)
    return result


def _target_components(data: dict[str, Any]) -> list[tuple[int, float, float]]:
    """Return (frequency, cosine coefficient, sine coefficient) triples."""
    if "target_components" in data:
        return [(int(component[0]), float(component[1]), float(component[2]))
                for component in data["target_components"]]
    return [(int(data["target_frequency"]), float(data["target_cosine"]), float(data["target_sine"]))]


def _make_dataset(data: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(data["seed"]))
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        x = rng.uniform(0.0, 1.0, size=int(data[f"{split}_size"]))
        arrays[f"{split}_x"] = x
        arrays[f"{split}_y"] = _target(x, data)
    return arrays


def _unitary(parameters: np.ndarray) -> np.ndarray:
    mapping = dict(zip(PARAMETER_NAMES, np.asarray(parameters, dtype=float)))
    return np.asarray(Operator(build_qft_ansatz(mapping)).data, dtype=np.complex128)


def predict(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute ideal <Z_1>; encode x on q0 and initialize q1 in |0>."""
    x = np.asarray(x, dtype=float).reshape(-1)
    states = np.zeros((x.size, 4), dtype=np.complex128)
    states[:, 0] = np.cos(np.pi * x)
    states[:, 1] = np.sin(np.pi * x)  # Qiskit basis index is little-endian: q0 is bit 0.
    evolved = states @ _unitary(parameters).T
    z1 = np.asarray([1.0, 1.0, -1.0, -1.0])
    return np.real(np.sum(np.abs(evolved) ** 2 * z1[None, :], axis=1))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y_true - np.mean(y_true))**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0}


def _spectrum(y: np.ndarray) -> np.ndarray:
    return np.fft.rfft(y) / len(y)


def _write_figures(run_dir: Path, history: list[dict[str, Any]], grid: np.ndarray,
                   target_grid: np.ndarray, pred_grid: np.ndarray, initial: np.ndarray,
                   final: np.ndarray, target: np.ndarray, classical: np.ndarray,
                   constant: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([r["step"] for r in history], [r["train_mse"] for r in history], label="train MSE")
    ax.plot([r["step"] for r in history], [r["validation_mse"] for r in history], label="validation MSE")
    ax.set(xlabel="Optimizer iteration", ylabel="MSE", yscale="log", title="MSE optimization history")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "loss.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # Target and known-k Fourier regression can coincide exactly. Draw the
    # continuous blue target first and green dashes over it to distinguish both.
    ax.plot(grid, target_grid, label="Target (ground truth)", color="royalblue",
            linewidth=2.6, alpha=.95, zorder=2)
    ax.plot(grid, classical, label="Fourier regression (known k; overlaps target)", color="forestgreen",
            linestyle=(0, (2, 2)), linewidth=2.0, zorder=3)
    ax.plot(grid, pred_grid, label="QNN prediction", color="darkorange",
            linestyle="--", linewidth=1.8, zorder=4)
    ax.axhline(float(np.mean(target)), color="gray", linestyle=":", label="test-target mean", zorder=1)
    ax.set(xlabel="x (period 1)", ylabel="Target / prediction", title="Function reconstruction on one period")
    ax.set_xlim(0, 1); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(run_dir / "function_fit.png", dpi=160); plt.close(fig)

    frequencies = np.arange(len(_spectrum(target_grid)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.stem(frequencies[:17], np.abs(_spectrum(target_grid)[:17]), linefmt="C0-", markerfmt="C0o", basefmt=" ", label="target")
    ax.stem(frequencies[:17], np.abs(_spectrum(pred_grid)[:17]), linefmt="C1--", markerfmt="C1x", basefmt=" ", label="QNN")
    ax.set(xlabel="Integer Fourier frequency k", ylabel="Amplitude", title="Period-one Fourier amplitude spectrum")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "spectrum.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(grid, np.abs(pred_grid - target_grid), color="C3")
    ax.set(xlabel="x (period 1)", ylabel="Absolute residual", title="QNN absolute error over one period")
    ax.set_xlim(0, 1); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(run_dir / "residual.png", dpi=160); plt.close(fig)

    fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
    positions = np.arange(len(PARAMETER_NAMES))
    qft_reference = QFT_PARAMETERS
    ax.bar(positions - .22, initial, width=.32, label="initial", color="steelblue")
    ax.bar(positions + .22, final, width=.32, label="final", color="darkorange")
    ax.scatter(positions, qft_reference, marker="D", s=45, color="forestgreen",
               edgecolor="white", linewidth=.6, label="canonical QFT parameters", zorder=5)
    ax.set_ylabel("Angle (radians)"); ax.set_title("Parameters: learned values and QFT reference")
    ax.grid(axis="y", alpha=.25); ax.legend(ncol=3, fontsize=8)
    circular_delta = (final - initial + np.pi) % (2 * np.pi) - np.pi
    delta_ax.bar(positions, circular_delta, color="slateblue", width=.55)
    delta_ax.axhline(0.0, color="forestgreen", linewidth=1.2, label="zero wrapped change")
    delta_ax.set_ylabel("Wrapped Δ (rad)"); delta_ax.set_title("Shortest signed angular change (modulo 2π)")
    delta_ax.set_xticks(positions, PARAMETER_NAMES, rotation=30, ha="right")
    delta_ax.grid(axis="y", alpha=.25); delta_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "parameters.png", dpi=160); plt.close(fig)


def run_function_mse(config_path: Path, seed: int, *, project_root: Path | None = None,
                     output_root: Path | None = None, data_seed: int | None = None) -> Path:
    config_path = config_path.resolve()
    project_root = (project_root or config_path.parent.parent).resolve()
    cfg = _load_config(config_path)
    exp, data_cfg, opt_cfg = cfg["experiment"], cfg["data"], cfg["optimizer"]
    if data_seed is not None:
        data_cfg["seed"] = int(data_seed)
    effective_config = config_path.read_text(encoding="utf-8")
    if data_seed is not None:
        lines = effective_config.splitlines()
        in_data_section = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_data_section = stripped == "[data]"
            elif in_data_section and stripped.startswith("seed") and "=" in line:
                indent = line[:len(line) - len(line.lstrip())]
                comment = f"  #{line.split('#', 1)[1]}" if "#" in line else ""
                lines[index] = f"{indent}seed = {int(data_cfg['seed'])}{comment}"
                break
        effective_config = "\n".join(lines) + "\n"
    effective_config_bytes = effective_config.encode("utf-8")
    data = _make_dataset(data_cfg)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, size=len(PARAMETER_NAMES))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_hash = hashlib.sha256(effective_config_bytes).hexdigest()[:8]
    run_id = f"{timestamp}_{exp['name']}_seed-{seed:04d}_{config_hash}"
    root = output_root or Path(cfg["output"]["root"])
    if not root.is_absolute(): root = project_root / root
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.toml").write_bytes(effective_config_bytes)
    np.savez_compressed(run_dir / "dataset.npz", **data)
    components = _target_components(data_cfg)
    target_manifest = {"target_offset": data_cfg["target_offset"],
                       "components": [{"frequency": k, "cosine": c, "sine": s}
                                      for k, c, s in components]}
    dataset_manifest = {
        "schema_version": "1.0", "generator": "qftlab.experiments.function_mse._make_dataset",
        "data_seed": int(data_cfg["seed"]), "encoding": "RY(2*pi*x)|0> on q0, x in [0,1), period 1",
        "readout": "ideal Pauli-Z expectation on q1",
        "target": target_manifest,
        "dataset_sizes": {split: int(data_cfg[f"{split}_size"])
                           for split in ("train", "validation", "test")},
        "arrays": {key: {"shape": list(v.shape), "dtype": str(v.dtype)} for key, v in data.items()},
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    resolved = json.loads(json.dumps(cfg))
    resolved["run"] = {"id": run_id, "seed": int(seed), "data_seed": int(data_cfg["seed"]),
                        "config_data_seed_overridden": data_seed is not None}
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    x_train, y_train = data["train_x"], data["train_y"]
    x_val, y_val = data["validation_x"], data["validation_y"]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    step = 0

    def objective(theta: np.ndarray) -> float:
        pred = predict(theta, x_train)
        return float(np.mean((pred - y_train) ** 2))

    def callback(theta: np.ndarray) -> None:
        nonlocal step
        step += 1
        row = {"step": step, "train_mse": objective(theta),
               "validation_mse": _metrics(y_val, predict(theta, x_val))["mse"],
               "elapsed_seconds": time.perf_counter() - started}
        row.update({f"parameter_{name}": float(value) for name, value in zip(PARAMETER_NAMES, theta)})
        history.append(row)

    # This is the sole training objective: supervised empirical MSE on (x, y) pairs.
    history.append({"step": 0, "train_mse": objective(initial),
                    "validation_mse": _metrics(y_val, predict(initial, x_val))["mse"],
                    "elapsed_seconds": 0.0,
                    **{f"parameter_{name}": float(value) for name, value in zip(PARAMETER_NAMES, initial)}})
    result = minimize(objective, initial, method="L-BFGS-B", callback=callback,
                      options={"maxiter": int(opt_cfg["max_iterations"]), "ftol": float(opt_cfg["ftol"]),
                               "gtol": float(opt_cfg["gtol"]), "maxfun": int(opt_cfg["max_iterations"]) * 30})
    final = np.asarray(result.x, dtype=float)
    test_pred = predict(final, data["test_x"])
    grid = np.arange(2048, dtype=float) / 2048
    target_grid = _target(grid, data_cfg)
    pred_grid = predict(final, grid)

    # Known-frequency Fourier regression is an informed reference using every target frequency.
    design_columns = [np.ones_like(x_train)]
    for frequency, _, _ in components:
        design_columns.extend([np.cos(2*np.pi*frequency*x_train), np.sin(2*np.pi*frequency*x_train)])
    design = np.column_stack(design_columns)
    coeff, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    x_test = data["test_x"]
    test_columns = [np.ones_like(x_test)]
    for frequency, _, _ in components:
        test_columns.extend([np.cos(2*np.pi*frequency*x_test), np.sin(2*np.pi*frequency*x_test)])
    test_design = np.column_stack(test_columns)
    classical_pred = test_design @ coeff
    constant_pred = np.full_like(data["test_y"], np.mean(y_train))
    grid_columns = [np.ones_like(grid)]
    for frequency, _, _ in components:
        grid_columns.extend([np.cos(2*np.pi*frequency*grid), np.sin(2*np.pi*frequency*grid)])
    classical_grid = np.column_stack(grid_columns) @ coeff

    learned_spectrum, target_spectrum = _spectrum(pred_grid), _spectrum(target_grid)
    abs_error = np.abs(pred_grid - target_grid)
    spec_size = min(17, len(target_spectrum))
    spectral_mse = float(np.mean(np.abs(learned_spectrum[:spec_size] - target_spectrum[:spec_size])**2))
    learned_peak = int(np.argmax(np.abs(learned_spectrum[1:spec_size])) + 1)
    target_peak = int(np.argmax(np.abs(target_spectrum[1:spec_size])) + 1)
    test_metrics = _metrics(data["test_y"], test_pred)
    summary = {
        "run_id": run_id, "status": "FINISHED", "comparison_group": exp["comparison_group"],
        "protocol_version": exp["protocol_version"], "objective": "training-set mean squared error only",
        "loss_excludes": ["process fidelity", "Frobenius/unitary alignment", "PINN/physics residual", "frequency labels"],
        "encoding": "RY(2*pi*x)|0> on q0; x in [0,1), period 1",
        "ansatz": cfg["model"]["ansatz"], "readout": "Pauli-Z expectation on q1",
        "seed": int(seed), "data_seed": int(data_cfg["seed"]), "optimizer": opt_cfg["name"],
        "dataset_sizes": {split: int(data_cfg[f"{split}_size"])
                          for split in ("train", "validation", "test")},
        "target_offset": float(data_cfg["target_offset"]),
        "target_components": [{"frequency": k, "cosine": c, "sine": s} for k, c, s in components],
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "elapsed_seconds": time.perf_counter() - started,
        "initial_train_mse": history[0]["train_mse"], "final_train_mse": objective(final),
        "validation_mse": _metrics(y_val, predict(final, x_val))["mse"],
        "test": test_metrics, "constant_baseline_test": _metrics(data["test_y"], constant_pred),
        "known_frequency_fourier_baseline_test": _metrics(data["test_y"], classical_pred),
        "spectral_coefficient_mse_k0_to_16": spectral_mse,
        "dense_grid_absolute_error_mae": float(np.mean(abs_error)),
        "dense_grid_absolute_error_max": float(np.max(abs_error)),
        "target_peak_frequency": target_peak, "predicted_peak_frequency": learned_peak,
        "frequency_peak_correct": learned_peak == target_peak,
        "qft_process_fidelity_diagnostic_only": process_fidelity(reference_unitary(), _unitary(final)),
        "initial_qft_process_fidelity_diagnostic_only": process_fidelity(reference_unitary(), _unitary(initial)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(run_dir / "evaluation_outputs.npz", test_x=x_test, test_y=data["test_y"],
                        test_prediction=test_pred, grid_x=grid, grid_target=target_grid,
                        grid_prediction=pred_grid, grid_fourier_baseline=classical_grid,
                        target_spectrum=target_spectrum, predicted_spectrum=learned_spectrum)
    (run_dir / "parameters.json").write_text(json.dumps({
        "initial": dict(zip(PARAMETER_NAMES, map(float, initial))),
        "final": dict(zip(PARAMETER_NAMES, map(float, final))),
    }, indent=2), encoding="utf-8")
    _write_figures(run_dir, history, grid, target_grid, pred_grid, initial, final,
                   data["test_y"], classical_grid, constant_pred)
    html = f'''<!doctype html><meta charset="utf-8"><title>{run_id}</title>
<h1>Periodic function regression: {run_id}</h1>
<p><b>Comparison group:</b> {exp['comparison_group']}<br><b>Training loss:</b> MSE only; no Frobenius/process-fidelity term or PINN residual.<br>
<b>Encoding:</b> RY(2πx)|0⟩ on qubit 0, x∈[0,1), period 1.<br><b>Readout:</b> ideal Pauli-Z expectation on qubit 1.</p>
<p><b>Absolute error over one period:</b> mean={summary['dense_grid_absolute_error_mae']:.6g}, max={summary['dense_grid_absolute_error_max']:.6g}; defined pointwise as |QNN prediction − target| on the 2048-point periodic grid.</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<img src="loss.png" width="49%"><img src="function_fit.png" width="49%"><br>
<img src="spectrum.png" width="49%"><img src="residual.png" width="49%"><br><img src="parameters.png" width="70%">'''
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    manifest = {
        "run_id": run_id, "status": "FINISHED", "created_at_utc": timestamp,
        "source_revision": _git_revision(project_root), "python": platform.python_version(),
        "packages": {name: _version(name) for name in ("numpy", "scipy", "qiskit", "matplotlib")},
        "config_sha256": hashlib.sha256(effective_config_bytes).hexdigest(),
        "comparison_group": exp["comparison_group"], "summary": "summary.json",
        "artifacts": ["dataset.npz", "dataset_manifest.json", "metrics.jsonl", "evaluation_outputs.npz",
                      "parameters.json", "loss.png", "function_fit.png", "spectrum.png", "residual.png",
                      "parameters.png", "report.html"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def refresh_function_mse_run(run_dir: Path) -> None:
    """Rebuild plots/report from a saved run without rerunning optimization."""
    run_dir = run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
    outputs = np.load(run_dir / "evaluation_outputs.npz")
    history = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    grid, target_grid = outputs["grid_x"], outputs["grid_target"]
    pred_grid = outputs["grid_prediction"]
    abs_error = np.abs(pred_grid - target_grid)
    summary["dense_grid_absolute_error_mae"] = float(np.mean(abs_error))
    summary["dense_grid_absolute_error_max"] = float(np.max(abs_error))
    initial = np.asarray([params["initial"][name] for name in PARAMETER_NAMES])
    final = np.asarray([params["final"][name] for name in PARAMETER_NAMES])
    _write_figures(run_dir, history, grid, target_grid, pred_grid, initial, final,
                   outputs["test_y"], outputs["grid_fourier_baseline"], outputs["test_y"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    html = f'''<!doctype html><meta charset="utf-8"><title>{summary['run_id']}</title>
<h1>Periodic function regression: {summary['run_id']}</h1>
<p><b>Comparison group:</b> {summary['comparison_group']}<br><b>Training loss:</b> MSE only; no Frobenius/process-fidelity term or PINN residual.<br>
<b>Encoding:</b> RY(2πx)|0⟩ on qubit 0, x∈[0,1), period 1.<br><b>Readout:</b> ideal Pauli-Z expectation on qubit 1.</p>
<p><b>Absolute error over one period:</b> mean={summary['dense_grid_absolute_error_mae']:.6g}, max={summary['dense_grid_absolute_error_max']:.6g}; pointwise |QNN prediction − target| on the 2048-point periodic grid.</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<img src="loss.png" width="49%"><img src="function_fit.png" width="49%"><br>
<img src="spectrum.png" width="49%"><img src="residual.png" width="49%"><br><img src="parameters.png" width="85%">'''
    (run_dir / "report.html").write_text(html, encoding="utf-8")
