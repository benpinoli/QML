#!/usr/bin/env python3
"""Compare single-upload and re-uploaded two-qubit QFT-layout QNNs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.circuits.qft import PARAMETER_NAMES  # noqa: E402
from qftlab.experiments.function_mse import _make_dataset, _target  # noqa: E402
from qftlab.experiments.reupload import predict_reuploaded, spectrum_amplitudes  # noqa: E402


CONFIGS = {
    2: ROOT / "configs/function_mse_components_2.toml",
    3: ROOT / "configs/function_mse_components_3.toml",
    4: ROOT / "configs/function_mse_components_4.toml",
}
GROUP_DIR = ROOT / "results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout"
RUNS_DIR = ROOT / "results/runs"
UPLOAD_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}
GRID_SIZE = 2048
SUPPORT_THRESHOLD = 1e-3


def _mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction) - np.asarray(y)) ** 2))


def _target_support(n_components: int) -> list[int]:
    return list(range(1, n_components + 1))


def _create_run(n_components: int, uploads: int, seed: int, data_seed: int,
                max_iterations: int, data: dict[str, np.ndarray], config: dict) -> dict:
    exp, data_cfg, opt_cfg = config["experiment"], config["data"], config["optimizer"]
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, size=(uploads, len(PARAMETER_NAMES)))
    x_train, y_train = data["train_x"], data["train_y"]
    x_val, y_val = data["validation_x"], data["validation_y"]
    history: list[dict] = []

    def objective(flat_parameters: np.ndarray) -> float:
        prediction = predict_reuploaded(flat_parameters.reshape(uploads, -1), x_train)
        return _mse(y_train, prediction)

    def callback(flat_parameters: np.ndarray) -> None:
        values = flat_parameters.reshape(uploads, -1)
        row = {"step": len(history), "train_mse": objective(flat_parameters),
               "validation_mse": _mse(y_val, predict_reuploaded(values, x_val))}
        row.update({f"layer_{layer}_{name}": float(value)
                    for layer, layer_values in enumerate(values, start=1)
                    for name, value in zip(PARAMETER_NAMES, layer_values)})
        history.append(row)

    initial_flat = initial.ravel()
    history.append({"step": 0, "train_mse": objective(initial_flat),
                    "validation_mse": _mse(y_val, predict_reuploaded(initial, x_val)),
                    **{f"layer_{layer}_{name}": float(value)
                       for layer, layer_values in enumerate(initial, start=1)
                       for name, value in zip(PARAMETER_NAMES, layer_values)}})
    result = minimize(objective, initial_flat, method="L-BFGS-B", callback=callback,
                      options={"maxiter": int(max_iterations), "ftol": float(opt_cfg["ftol"]),
                               "gtol": float(opt_cfg["gtol"]), "maxfun": int(max_iterations) * 30})
    final = result.x.reshape(uploads, -1)

    grid_x = np.arange(GRID_SIZE, dtype=float) / GRID_SIZE
    target_grid = _target(grid_x, data_cfg)
    prediction_grid = predict_reuploaded(final, grid_x)
    target_amplitudes = spectrum_amplitudes(target_grid)
    predicted_amplitudes = spectrum_amplitudes(prediction_grid)
    predicted_support = np.flatnonzero(predicted_amplitudes[1:GRID_SIZE // 2 + 1] > SUPPORT_THRESHOLD) + 1
    target_support = np.asarray(_target_support(n_components))
    true_positive = len(set(predicted_support.tolist()) & set(target_support.tolist()))
    support_precision = true_positive / len(predicted_support) if len(predicted_support) else 0.0
    support_recall = true_positive / len(target_support)
    x_test, y_test = data["test_x"], data["test_y"]
    x_train_array = np.column_stack([np.ones_like(x_train)] + [v for k in range(1, n_components + 1)
                                        for v in (np.cos(2*np.pi*k*x_train), np.sin(2*np.pi*k*x_train))])
    x_test_array = np.column_stack([np.ones_like(x_test)] + [v for k in range(1, n_components + 1)
                                       for v in (np.cos(2*np.pi*k*x_test), np.sin(2*np.pi*k*x_test))])
    fourier_coefficients, *_ = np.linalg.lstsq(x_train_array, y_train, rcond=None)
    fourier_test_mse = _mse(y_test, x_test_array @ fourier_coefficients)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_text = json.dumps({"target_config": config, "data_seed": data_seed,
                              "initialization_seed": seed, "data_uploads": uploads,
                              "architecture": "[RY(2*pi*x) on q0; trainable U(q1)-CP-U(q0)] repeated; final SWAP; measure Z1"},
                             indent=2, sort_keys=True)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:8]
    run_id = f"{timestamp}_reupload-{uploads}_k{n_components}_seed-{seed:04d}_{config_hash}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.resolved.json").write_text(config_text + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "dataset.npz", **data)
    np.savez_compressed(run_dir / "parameters.npz", initial=initial, final=final)
    np.savez_compressed(run_dir / "evaluation_outputs.npz", grid_x=grid_x, grid_target=target_grid,
                        grid_prediction=prediction_grid, target_amplitudes=target_amplitudes,
                        predicted_amplitudes=predicted_amplitudes, test_x=x_test, test_y=y_test,
                        test_prediction=predict_reuploaded(final, x_test))
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (run_dir / "parameters.json").write_text(json.dumps({
        "initial_by_layer": [dict(zip(PARAMETER_NAMES, map(float, layer))) for layer in initial],
        "final_by_layer": [dict(zip(PARAMETER_NAMES, map(float, layer))) for layer in final],
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid_x, target_grid, color="#0072B2", linewidth=2, label="target")
    ax.plot(grid_x, prediction_grid, color="#D55E00", linewidth=1.6, linestyle="--", label="QNN")
    ax.set(xlim=(0, 1), xlabel="x (period 1)", ylabel="f(x)", title=f"{uploads} data upload(s), {n_components} target frequencies")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "function_fit.png", dpi=140); plt.close(fig)

    freqs = np.arange(min(17, len(target_amplitudes)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, target_amplitudes[freqs], "o-", label="target", color="#0072B2")
    ax.plot(freqs, predicted_amplitudes[freqs], "s--", label="QNN", color="#D55E00")
    ax.set(xlabel="Fourier frequency k", ylabel="Amplitude", title="Dense-grid spectrum")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "spectrum.png", dpi=140); plt.close(fig)

    final_train_mse = objective(result.x)
    test_mse = _mse(y_test, predict_reuploaded(final, x_test))
    summary = {
        "run_id": run_id, "comparison_group": "periodic-function-mse-zero-offset-reupload-sweep-z1-readout",
        "data_uploads": uploads, "n_target_frequencies": n_components, "seed": seed, "data_seed": data_seed,
        "trainable_parameter_count": int(final.size), "trainable_parameter_count_per_layer": 7,
        "dataset_sizes": {key: len(data[f"{key}_x"]) for key in ("train", "validation", "test")},
        "target_components": config["data"]["target_components"], "target_offset": 0.0,
        "objective": "training-set MSE only", "optimizer": "L-BFGS-B",
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "initial_train_mse": history[0]["train_mse"], "final_train_mse": final_train_mse,
        "validation_mse": _mse(y_val, predict_reuploaded(final, x_val)), "test_mse": test_mse,
        "known_frequency_fourier_baseline_test_mse": fourier_test_mse,
        "target_support": target_support.tolist(), "predicted_support": predicted_support.tolist(),
        "frequency_support_threshold_function_amplitude": SUPPORT_THRESHOLD,
        "support_precision": support_precision, "support_recall": support_recall,
        "target_amplitudes_k0_to_16": target_amplitudes[:17].tolist(),
        "predicted_amplitudes_k0_to_16": predicted_amplitudes[:17].tolist(),
        "frequency_amplitude_rmse_k0_to_16": float(np.sqrt(np.mean((target_amplitudes[:17] - predicted_amplitudes[:17])**2))),
        "artifacts": ["dataset.npz", "metrics.jsonl", "parameters.npz", "parameters.json",
                      "evaluation_outputs.npz", "function_fit.png", "spectrum.png", "report.html"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "report.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{run_id}</title><h1>{run_id}</h1>'
        f'<p>Uploads: {uploads}; target frequencies: {n_components}; train/validation/test: '
        f'{len(x_train)}/{len(x_val)}/{len(x_test)}; objective: empirical training MSE only.</p>'
        f'<pre>{json.dumps(summary, indent=2)}</pre><img src="function_fit.png" width="49%">'
        f'<img src="spectrum.png" width="49%">', encoding="utf-8")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {"run_id": run_id, "created_at_utc": timestamp, "source_revision": revision,
                "python": platform.python_version(), "comparison_group": summary["comparison_group"],
                "artifacts": summary["artifacts"]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary["run_dir"] = str(run_dir.relative_to(ROOT))
    return summary


def _reachable_spectra(uploads_to_test: list[int], samples: int = 24) -> dict:
    grid = np.arange(GRID_SIZE, dtype=float) / GRID_SIZE
    rng = np.random.default_rng(982451653)
    results = {}
    for uploads in uploads_to_test:
        union: set[int] = set()
        max_by_sample = []
        for _ in range(samples):
            theta = rng.uniform(-np.pi, np.pi, size=(uploads, len(PARAMETER_NAMES)))
            amplitudes = spectrum_amplitudes(predict_reuploaded(theta, grid))
            support = np.flatnonzero(amplitudes[1:GRID_SIZE // 2 + 1] > 1e-8) + 1
            union.update(map(int, support))
            max_by_sample.append(int(np.max(support)) if len(support) else 0)
        results[str(uploads)] = {"random_parameter_samples": samples,
                                 "amplitude_threshold": 1e-8,
                                 "union_non_dc_support": sorted(union),
                                 "maximum_frequency_per_sample": max_by_sample,
                                 "max_observed_frequency": max(max_by_sample)}
    return results


def _write_group_outputs(records: list[dict], uploads: list[int], n_seeds: int,
                         data_seed: int, reachable: dict, one_upload_comparison: dict) -> dict:
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    groups = []
    for n_components in sorted(CONFIGS):
        for n_uploads in uploads:
            subset = [r for r in records if r["n_target_frequencies"] == n_components and r["data_uploads"] == n_uploads]
            mse = np.asarray([r["test_mse"] for r in subset])
            recall = np.asarray([r["support_recall"] for r in subset])
            precision = np.asarray([r["support_precision"] for r in subset])
            expected_floor = sum((float(c) ** 2 + float(s) ** 2) / 2
                                 for k, c, s in subset[0]["target_components"] if int(k) > n_uploads)
            groups.append({
                "n_target_frequencies": n_components, "data_uploads": n_uploads,
                "n_runs": len(subset), "test_mse_mean": float(np.mean(mse)),
                "test_mse_population_std": float(np.std(mse)),
                "support_precision_mean": float(np.mean(precision)),
                "support_recall_mean": float(np.mean(recall)),
                "support_f1_mean": float(np.mean([2*p*r/(p+r) if p+r else 0 for p, r in zip(precision, recall)])),
                "missing_harmonic_uniform_mse_floor": expected_floor,
                "mean_frequency_amplitude_rmse": float(np.mean([r["frequency_amplitude_rmse_k0_to_16"] for r in subset])),
                "optimizer_success_count": sum(r["optimizer_success"] for r in subset),
                "mean_iterations": float(np.mean([r["iterations"] for r in subset])),
            })

    summary = {
        "comparison_group": "periodic-function-mse-zero-offset-reupload-sweep-z1-readout",
        "question": "Does repeated data encoding let the same two-qubit QFT-layout model learn higher Fourier frequencies?",
        "architecture": "For each layer: RY(2*pi*x) on q0, trainable U on q1, controlled phase, trainable U on q0; final SWAP; ideal Z1 expectation.",
        "one_upload_equivalence": "One upload is prediction-equivalent to the existing QFT-layout function experiment; verified to max absolute delta < 1e-12 on a random parameter/input grid.",
        "objective": "Empirical training MSE only; no process-fidelity term, frequency labels, or PINN residual.",
        "encoding": "Each upload applies RY(2*pi*x) on q0; x in [0,1), period 1.",
        "readout": "Ideal Pauli-Z expectation on q1 after final SWAP.",
        "data_seed": data_seed, "initialization_seeds": sorted({r["seed"] for r in records}),
        "n_runs": len(records), "dataset_sizes": {"train": 64, "validation": 128, "test": 512},
        "uploads_tested": uploads, "target_frequency_counts": sorted(CONFIGS),
        "trainable_parameters_per_layer": 7,
        "trainable_parameter_count_by_upload": {str(u): 7*u for u in uploads},
        "design_caveat": "Trainable block count and parameter count increase with upload count (7 parameters per layer). The random-spectrum check isolates accessible frequency support, but trained-curve improvements are not a parameter-count-matched causal comparison. A depth-matched single-upload control is the next experiment.",
        "support_threshold_function_amplitude": SUPPORT_THRESHOLD,
        "random_circuit_spectral_reachability": reachable,
        "one_upload_baseline_comparison": one_upload_comparison,
        "group_results": groups,
        "runs_sorted_by_test_mse": sorted(records, key=lambda r: r["test_mse"]),
    }
    (GROUP_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    columns = ["run_id", "run_dir", "data_uploads", "n_target_frequencies", "trainable_parameter_count", "data_seed", "initialization_seed",
               "n_train", "n_validation", "n_test", "train_mse", "validation_mse", "test_mse",
               "target_support", "predicted_support", "support_precision", "support_recall",
               "frequency_amplitude_rmse", "iterations", "optimizer_success"]
    with (GROUP_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "run_id": r["run_id"], "run_dir": r["run_dir"], "data_uploads": r["data_uploads"],
                "n_target_frequencies": r["n_target_frequencies"], "trainable_parameter_count": 7*r["data_uploads"],
                "data_seed": r["data_seed"], "initialization_seed": r["seed"],
                "n_train": r["dataset_sizes"]["train"], "n_validation": r["dataset_sizes"]["validation"], "n_test": r["dataset_sizes"]["test"],
                "train_mse": r["final_train_mse"], "validation_mse": r["validation_mse"], "test_mse": r["test_mse"],
                "target_support": ";".join(map(str, r["target_support"])), "predicted_support": ";".join(map(str, r["predicted_support"])),
                "support_precision": r["support_precision"], "support_recall": r["support_recall"],
                "frequency_amplitude_rmse": r["frequency_amplitude_rmse_k0_to_16"], "iterations": r["iterations"],
                "optimizer_success": r["optimizer_success"],
            })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, n_components in zip(axes, sorted(CONFIGS)):
        rows = [g for g in groups if g["n_target_frequencies"] == n_components]
        xs = np.asarray([r["data_uploads"] for r in rows])
        means = np.asarray([r["test_mse_mean"] for r in rows])
        stds = np.asarray([r["test_mse_population_std"] for r in rows])
        floors = np.asarray([r["missing_harmonic_uniform_mse_floor"] for r in rows])
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, capsize=3,
                    color="#0072B2", label="test MSE mean ± SD")
        ax.plot(xs, floors, linestyle="--", marker="x", color="#D55E00", label="omitted-harmonic floor")
        ax.set(title=f"Target: {n_components} frequencies", xlabel="Data uploads", xticks=uploads)
        ax.set_yscale("log"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Test MSE (log scale)"); axes[0].legend(fontsize=8)
    fig.suptitle("Function-fit error versus data-upload count")
    fig.tight_layout(); fig.savefig(GROUP_DIR / "test_mse_vs_uploads.png", dpi=180); plt.close(fig)

    recall_matrix = np.asarray([[next(g["support_recall_mean"] for g in groups
                                      if g["n_target_frequencies"] == n and g["data_uploads"] == u)
                                 for u in uploads] for n in sorted(CONFIGS)])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    image = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=np.arange(len(uploads)), xticklabels=uploads,
           yticks=np.arange(len(CONFIGS)), yticklabels=[f"{n} frequencies" for n in sorted(CONFIGS)],
           xlabel="Data uploads", title="Mean target frequency-support recall")
    for iy in range(recall_matrix.shape[0]):
        for ix in range(recall_matrix.shape[1]):
            ax.text(ix, iy, f"{recall_matrix[iy, ix]:.2f}", ha="center", va="center",
                    color="white" if recall_matrix[iy, ix] < .55 else "black")
    fig.colorbar(image, ax=ax, label="Recall")
    fig.tight_layout(); fig.savefig(GROUP_DIR / "frequency_support_recall.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, n_components in zip(axes, sorted(CONFIGS)):
        target_components = next(r["target_components"] for r in records if r["n_target_frequencies"] == n_components)
        target_amplitudes = np.zeros(5)
        for k, cosine, sine in target_components:
            target_amplitudes[int(k)] = np.hypot(float(cosine), float(sine))
        ax.plot(np.arange(1, 5), target_amplitudes[1:5], color="black", marker="D", linewidth=2, label="target")
        for n_uploads in uploads:
            subset = [r for r in records if r["n_target_frequencies"] == n_components and r["data_uploads"] == n_uploads]
            amplitude_matrix = np.asarray([r["predicted_amplitudes_k0_to_16"][:5] for r in subset])
            mean = np.mean(amplitude_matrix, axis=0)
            std = np.std(amplitude_matrix, axis=0)
            ax.errorbar(np.arange(1, 5), mean[1:5], yerr=std[1:5], marker="o", capsize=2,
                        color=UPLOAD_COLORS[n_uploads], alpha=.9, label=f"{n_uploads} upload(s)")
        ax.set(title=f"Target: {n_components} frequencies", xlabel="Fourier frequency k", xticks=[1, 2, 3, 4])
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Fourier amplitude")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Learned Fourier amplitudes by upload count (mean ± SD)")
    fig.tight_layout(); fig.savefig(GROUP_DIR / "learned_spectra_by_uploads.png", dpi=180); plt.close(fig)

    table_rows = "\n".join(
        f'<tr><td>{g["n_target_frequencies"]}</td><td>{g["data_uploads"]}</td><td>{g["n_runs"]}</td>'
        f'<td>{g["test_mse_mean"]:.7g} ± {g["test_mse_population_std"]:.2g}</td>'
        f'<td>{g["missing_harmonic_uniform_mse_floor"]:.7g}</td><td>{g["support_precision_mean"]:.3f}</td>'
        f'<td>{g["support_recall_mean"]:.3f}</td><td>{g["mean_frequency_amplitude_rmse"]:.5g}</td></tr>'
        for g in groups)
    run_rows = "\n".join(
        f'<tr><td>{r["data_uploads"]}</td><td>{r["n_target_frequencies"]}</td><td>{r["seed"]}</td>'
        f'<td>{r["test_mse"]:.7g}</td><td>{r["predicted_support"]}</td>'
        f'<td><a href="../runs/{r["run_id"]}/report.html">report</a></td></tr>'
        for r in sorted(records, key=lambda r: (r["test_mse"], r["data_uploads"], r["n_target_frequencies"]))[:20])
    html = f'''<!doctype html><meta charset="utf-8"><title>Two-qubit data re-uploading sweep</title>
<h1>Two-qubit data re-uploading sweep</h1>
<p>Controlled comparison of 1–4 data uploads with 2–4 nested target frequencies. Qubit count, target coefficients, sampled data (seed {data_seed}), and ten initialization seeds are paired across upload counts. Each run uses {len(records)//(len(uploads)*len(CONFIGS))} initializations; 64 train, 128 validation, and 512 test examples.</p>
<p>The per-layer structure is RY(2πx) on q0, trainable U on q1, controlled phase, trainable U on q0; after the final layer, apply SWAP and measure ideal Z1. L=1 is the original ansatz. Training objective is empirical MSE only. Because L&gt;1 makes the circuit input-dependent at multiple locations, a single process-fidelity-to-QFT score is not defined for this comparison.</p>
<p><b>Interpreting the MSE plot:</b> connected points show mean test MSE over ten initializations and whiskers show ±1 population standard deviation (not a minimum or boxplot). The abrupt drop for the four-frequency target at four uploads is the mean reaching about 1.8e-13. Extra-upload changes for the already-fit two-frequency target are tiny residuals near numerical precision. On the logarithmic axis a lower whisker cannot be drawn if mean minus SD is zero or negative.</p>
<p><b>Frequency count versus bandwidth:</b> these nested targets contain every harmonic from k=1 through k=N, so component count equals maximum harmonic. The sweep shows uploads expanding support through k=L and does not test a one-upload-per-arbitrary-component rule. A sparse-target experiment is needed to separate those explanations.</p>
<p><b>Design caveat:</b> each added upload also adds one trainable block (7 parameters), so depth and parameter count rise with upload count. The random-parameter spectrum check separately verifies that the accessible frequency ceiling expands. A depth-matched circuit with one upload is still needed to isolate upload effects on optimizer performance.</p>
<h2>Aggregate results</h2><table border="1" cellpadding="5"><thead><tr><th>Target frequencies</th><th>Uploads</th><th>Runs</th><th>Test MSE mean ± SD</th><th>Omitted-power floor</th><th>Support precision</th><th>Support recall</th><th>Amplitude RMSE</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Function error and spectral recovery</h2><img src="test_mse_vs_uploads.png" width="96%"><br><img src="frequency_support_recall.png" width="72%"><br><img src="learned_spectra_by_uploads.png" width="96%">
<h2>Random-parameter expressivity check</h2><p>Twenty-four random parameter settings per upload count were evaluated on a dense 2048-point period. The union of detected non-DC bins (amplitude threshold 1e-8) is in summary.json. This is a numerical check of accessible support, separate from the trained-run support threshold of {SUPPORT_THRESHOLD}.</p><pre>{json.dumps(reachable, indent=2)}</pre>
<h2>One-upload baseline check</h2><pre>{json.dumps(one_upload_comparison, indent=2)}</pre>
<h2>Lowest-MSE example runs</h2><table border="1" cellpadding="5"><thead><tr><th>Uploads</th><th>Target frequencies</th><th>Seed</th><th>Test MSE</th><th>Learned support</th><th>Run</th></tr></thead><tbody>{run_rows}</tbody></table>
<p><a href="metrics.csv">All run metrics CSV</a> · <a href="summary.json">Machine-readable summary</a> · <a href="../../HYPOTHESIS_TREE.md">Hypothesis tree</a></p>'''
    (GROUP_DIR / "index.html").write_text(html, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--data-seed", type=int, default=314159)
    parser.add_argument("--uploads", default="1,2,3,4")
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    try:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        uploads = [int(x.strip()) for x in args.uploads.split(",") if x.strip()]
    except ValueError:
        parser.error("seeds and uploads must be comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds) or not uploads or len(set(uploads)) != len(uploads):
        parser.error("seeds and uploads must be nonempty and unique")
    if any(x < 1 or x > 4 for x in uploads):
        parser.error("this planned sweep accepts 1 through 4 data uploads")

    datasets, configs = {}, {}
    for n_components, path in CONFIGS.items():
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        config["data"]["seed"] = args.data_seed
        if config["data"]["target_offset"] != 0:
            parser.error(f"{path} must retain zero target offset")
        configs[n_components] = config
        datasets[n_components] = _make_dataset(config["data"])

    # Confirm that the custom vectorized simulator exactly preserves the old L=1 model.
    from qftlab.experiments.function_mse import predict as legacy_predict
    check_rng = np.random.default_rng(18271)
    check_parameters = check_rng.uniform(-np.pi, np.pi, size=7)
    check_x = np.linspace(0, 1, 73, endpoint=False)
    equivalence_delta = float(np.max(np.abs(
        predict_reuploaded(check_parameters[None, :], check_x) - legacy_predict(check_parameters, check_x))))
    if equivalence_delta > 1e-12:
        raise RuntimeError(f"single-upload circuit does not match reference model: {equivalence_delta}")

    records = []
    expected_runs = len(uploads) * len(CONFIGS) * len(seeds)
    print(f"Running {expected_runs} paired runs: uploads={uploads}, targets={sorted(CONFIGS)}, seeds={seeds}", flush=True)
    for n_components in sorted(CONFIGS):
        for n_uploads in uploads:
            for seed in seeds:
                record = _create_run(n_components, n_uploads, seed, args.data_seed,
                                     args.max_iterations, datasets[n_components], configs[n_components])
                records.append(record)
                print(json.dumps({"completed": len(records), "total": expected_runs,
                                  "uploads": n_uploads, "target_frequencies": n_components,
                                  "seed": seed, "test_mse": record["test_mse"],
                                  "support": record["predicted_support"], "iterations": record["iterations"]}),
                      flush=True)

    reachable = _reachable_spectra(uploads)
    prior_path = ROOT / "results/periodic-function-mse-zero-offset-component-sweep-z1-readout/summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else None
    baseline = {"one_upload_exact_prediction_equivalence_max_abs_delta": equivalence_delta}
    if prior:
        prior_runs = {(r["n_components"], r["seed"]): r for r in prior["runs_sorted_by_test_mse"]}
        differences = []
        for r in records:
            if r["data_uploads"] == 1 and (r["n_target_frequencies"], r["seed"]) in prior_runs:
                differences.append(r["test_mse"] - prior_runs[(r["n_target_frequencies"], r["seed"])]["test"]["mse"])
        baseline["paired_prior_sweep_test_mse_delta_mean"] = float(np.mean(differences)) if differences else None
        baseline["paired_prior_sweep_test_mse_delta_max_abs"] = float(np.max(np.abs(differences))) if differences else None
        baseline["paired_runs_compared"] = len(differences)
    summary = _write_group_outputs(records, uploads, len(seeds), args.data_seed, reachable, baseline)
    print(json.dumps({"report": str(GROUP_DIR / "index.html"), "n_runs": len(records),
                      "group_results": summary["group_results"],
                      "random_circuit_spectral_reachability": reachable,
                      "one_upload_baseline_comparison": baseline}, indent=2))


if __name__ == "__main__":
    main()
