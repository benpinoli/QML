"""End-to-end QFT recovery experiment and portable run record."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qftlab import __version__
from qftlab.circuits.qft import PARAMETER_NAMES, QFT_PARAMETERS, reference_unitary
from qftlab.data import make_frequency_dataset, make_qft_dataset
from qftlab.evaluation import evaluate_qft
from qftlab.objectives import process_fidelity
from qftlab.reporting import write_report
from qftlab.training import optimize_unitary, parameters_to_unitary


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_fingerprint(project_root: Path) -> str:
    digest = hashlib.sha256()
    files = [project_root / "pyproject.toml"]
    files.extend((project_root / "src").rglob("*.py"))
    files.extend((project_root / "scripts").rglob("*.py"))
    files.extend((project_root / "configs").rglob("*.toml"))
    for path in sorted((path for path in files if path.is_file()), key=lambda item: str(item.relative_to(project_root))):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _log_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


class _MLflowLogger:
    def __init__(self, config: dict[str, Any], run_name: str, tags: dict[str, str]):
        tracking = config["tracking"]
        mode = tracking.get("mode", "auto")
        uri_env = tracking.get("tracking_uri_env", "MLFLOW_TRACKING_URI")
        uri = os.environ.get(uri_env)
        if mode == "off":
            self.mlflow = None
            return
        if mode == "required" and not uri:
            raise RuntimeError(f"Tracking mode is required but {uri_env} is not set.")
        if mode == "auto" and not uri:
            self.mlflow = None
            return
        try:
            import mlflow
        except ImportError as exc:
            if mode == "required":
                raise RuntimeError("MLflow is required by this config. Install with `pip install -e '.[tracking]'`.") from exc
            self.mlflow = None
            return
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(tracking.get("experiment_name", config["experiment"]["name"]))
        mlflow.start_run(run_name=run_name, tags=tags)
        self.mlflow = mlflow
        self.run_id = mlflow.active_run().info.run_id

    @property
    def enabled(self) -> bool:
        return self.mlflow is not None

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            safe = {str(k): str(v)[:500] for k, v in params.items()}
            self.mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.enabled:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifacts(self, directory: Path) -> None:
        if self.enabled:
            self.mlflow.log_artifacts(str(directory))

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.enabled:
            self.mlflow.set_tags(tags)

    def end(self, status: str) -> None:
        if self.enabled:
            self.mlflow.end_run(status="FINISHED" if status == "FINISHED" else "FAILED")


def load_toml(path: Path) -> dict[str, Any]:
    import tomllib
    with path.open("rb") as stream:
        return tomllib.load(stream)


def run_qft_recovery(config_path: Path, *, seed: int | None = None,
                     output_root: Path | None = None, project_root: Path | None = None) -> Path:
    config_path = config_path.resolve()
    project_root = (project_root or config_path.parent.parent).resolve()
    config = load_toml(config_path)
    experiment = config["experiment"]
    optimizer = config["optimizer"]
    if optimizer["name"].upper() != "L-BFGS-B":
        raise ValueError(f"This baseline currently supports optimizer.name='L-BFGS-B', got {optimizer['name']!r}.")
    data_config = config["data"]
    circuit_config = config["circuit"]
    tracking_config = config["tracking"]
    output_config = config["output"]
    run_seed = int(seed if seed is not None else experiment.get("initialization_seed", data_config["seed"]))
    data_seed = int(data_config["seed"])
    init_scale = float(circuit_config.get("initialization_scale", 1.0))
    rng = np.random.default_rng(run_seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
    slug = experiment["name"].replace("/", "-").replace(" ", "-")
    run_id = f"{timestamp}_{slug}_seed-{run_seed:04d}_{config_hash}"
    root = output_root or Path(output_config["root"])
    if not root.is_absolute():
        root = project_root / root
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.toml")

    initial = QFT_PARAMETERS + rng.uniform(-np.pi, np.pi, size=len(QFT_PARAMETERS)) * init_scale
    target = reference_unitary()
    eval_data = make_qft_dataset(
        data_seed, int(data_config["train_size"]), int(data_config["validation_size"]),
        int(data_config["test_size"]), target,
    )
    freq_data = make_frequency_dataset(target)
    np.savez_compressed(run_dir / "dataset.npz", **eval_data, **freq_data)
    (run_dir / "dataset_manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "generator": "qftlab.data.synthetic",
        "data_seed": data_seed, "state_dimension": 4,
        "statevector_basis_order": "Qiskit little-endian qubit order; amplitudes indexed by |q1 q0> basis integer",
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in {**eval_data, **freq_data}.items()},
    }, indent=2), encoding="utf-8")

    effective_config = json.loads(json.dumps(config))
    effective_config["run"] = {"id": run_id, "seed": run_seed, "data_seed": data_seed,
                                "source_config": config_path.name}
    (run_dir / "config.resolved.json").write_text(json.dumps(effective_config, indent=2), encoding="utf-8")
    tags = {
        "protocol_version": str(experiment.get("protocol_version", "unknown")),
        "ansatz": "qft2_trainable_u_cp_u_swap",
        "comparison_group": str(experiment.get("comparison_group", experiment["name"])),
        "execution": "slurm" if "SLURM_JOB_ID" in os.environ else "local",
        "seed": str(run_seed),
    }
    if os.environ.get("SLURM_JOB_ID"):
        tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        tags["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        tags["slurm_cluster"] = os.environ.get("SLURM_CLUSTER_NAME", "")

    logger = None
    try:
        logger = _MLflowLogger(config, run_id, tags)
        if logger.enabled:
            tags["mlflow_run_id"] = logger.run_id
            logger.set_tags(tags)
        logger.log_params({
            "protocol_version": experiment.get("protocol_version", "unknown"),
            "ansatz": tags["ansatz"], "optimizer": optimizer["name"], "max_iterations": optimizer["max_iterations"],
            "seed": run_seed, "data_seed": data_seed, "initialization_scale": init_scale,
            "train_size": data_config["train_size"], "validation_size": data_config["validation_size"],
            "test_size": data_config["test_size"], "ftol": optimizer["ftol"], "gtol": optimizer["gtol"],
            "max_line_search_steps": optimizer.get("max_line_search_steps", 30),
        })
    except Exception:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "experiment": experiment["name"], "status": "FAILED",
            "created_at_utc": timestamp, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_revision": _git_revision(project_root), "source_fingerprint": _source_fingerprint(project_root),
            "seeds": {"initialization": run_seed, "data": data_seed},
            "tracking_setup_failed": True,
        }, indent=2), encoding="utf-8")
        if logger is not None:
            try:
                logger.end("FAILED")
            except Exception:
                pass
        raise

    history: list[dict[str, Any]] = []
    status = "FAILED"
    training_started = time.perf_counter()
    try:
        train_inputs = eval_data["train_inputs"]
        train_targets = eval_data["train_targets"]
        val_inputs = eval_data["validation_inputs"]
        val_targets = eval_data["validation_targets"]

        def record(step: int, values: np.ndarray, loss: float) -> None:
            unitary = parameters_to_unitary(values)
            train_scores = np.abs(np.sum(train_targets.conj() * (train_inputs @ unitary.T), axis=1)) ** 2
            val_scores = np.abs(np.sum(val_targets.conj() * (val_inputs @ unitary.T), axis=1)) ** 2
            row = {
                "step": int(step), "loss": float(loss), "process_fidelity": float(1.0 - loss),
                "train_state_fidelity_mean": float(np.mean(train_scores)),
                "validation_state_fidelity_mean": float(np.mean(val_scores)),
                "elapsed_seconds": float(time.perf_counter() - training_started),
            }
            row.update({f"parameter_{name}": float(value) for name, value in zip(PARAMETER_NAMES, values)})
            history.append(row)
            logger.log_metrics({k: v for k, v in row.items() if k != "step"}, step=step)

        record(0, initial, float(1.0 - process_fidelity(target, parameters_to_unitary(initial))))
        result = optimize_unitary(
            target, initial, max_iterations=int(optimizer["max_iterations"]),
            ftol=float(optimizer["ftol"]), gtol=float(optimizer["gtol"]),
            max_line_search_steps=int(optimizer.get("max_line_search_steps", 30)), callback=record,
        )
        final_parameters = np.asarray(result.x, dtype=float)
        final_unitary = parameters_to_unitary(final_parameters)
        validation_eval = evaluate_qft(target, final_unitary, val_inputs, val_targets,
                                       freq_data["inputs"], freq_data["target_probabilities"])
        # The test split is evaluated only once, after optimization has ended.
        test_eval = evaluate_qft(target, final_unitary, eval_data["test_inputs"], eval_data["test_targets"],
                                 freq_data["inputs"], freq_data["target_probabilities"])
        summary = {
            "run_id": run_id, "status": "FINISHED", "optimizer": optimizer["name"],
            "protocol_version": experiment.get("protocol_version", "unknown"),
            "comparison_group": experiment.get("comparison_group", experiment["name"]),
            "ansatz": tags["ansatz"], "seed": run_seed, "data_seed": data_seed,
            "initialization_scale": init_scale, "max_iterations": int(optimizer["max_iterations"]),
            "train_size": int(data_config["train_size"]), "validation_size": int(data_config["validation_size"]),
            "test_size": int(data_config["test_size"]), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message), "initial_process_fidelity": float(1-history[0]["loss"]),
            "elapsed_seconds": float(time.perf_counter() - training_started),
            "qft_process_fidelity": float(test_eval["process_fidelity"]),
            "test_frequency_accuracy": float(test_eval["frequency_accuracy"]),
            "validation_state_fidelity_mean": float(validation_eval["heldout_state_fidelity_mean"]),
            "validation_output_total_variation_mean": float(validation_eval["heldout_output_total_variation_mean"]),
            "test_state_fidelity_mean": float(test_eval["heldout_state_fidelity_mean"]),
            "test_state_fidelity_min": float(test_eval["heldout_state_fidelity_min"]),
            "test_output_total_variation_mean": float(test_eval["heldout_output_total_variation_mean"]),
            "frequency_accuracy": float(test_eval["frequency_accuracy"]),
            "frequency_cross_entropy": float(test_eval["frequency_cross_entropy"]),
            "frequency_total_variation_mean": float(test_eval["frequency_total_variation_mean"]),
        }
        parameter_dict = dict(zip(PARAMETER_NAMES, final_parameters.tolist()))
        (run_dir / "parameters").mkdir()
        (run_dir / "parameters" / "initial.json").write_text(json.dumps(dict(zip(PARAMETER_NAMES, initial.tolist())), indent=2))
        (run_dir / "parameters" / "final.json").write_text(json.dumps(parameter_dict, indent=2))
        (run_dir / "parameters" / "target_qft.json").write_text(json.dumps(dict(zip(PARAMETER_NAMES, QFT_PARAMETERS.tolist())), indent=2))
        np.savez_compressed(
            run_dir / "evaluation_outputs.npz",
            test_inputs=eval_data["test_inputs"], test_targets=eval_data["test_targets"],
            test_predictions=eval_data["test_inputs"] @ final_unitary.T,
            test_frequency_probabilities=test_eval["frequency_probabilities"],
            test_frequency_targets=freq_data["target_probabilities"],
            test_frequency_predictions=test_eval["frequency_predictions"],
            parameter_history=np.asarray([initial] + [[row[f"parameter_{name}"] for name in PARAMETER_NAMES]
                                                       for row in history[1:]]),
        )
        _log_jsonl(run_dir / "metrics.jsonl", history)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(run_dir, history, freq_data["target_probabilities"],
                     test_eval["frequency_probabilities"], parameter_dict, summary)
        logger.log_metrics({k: v for k, v in summary.items() if isinstance(v, (float, int)) and k not in {"seed", "iterations", "function_evaluations"}}, step=len(history))
        status = "FINISHED"
        return run_dir
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _log_jsonl(run_dir / "metrics.jsonl", history)
        raise
    finally:
        manifest = {
            "run_id": run_id, "experiment": experiment["name"],
            "protocol_version": experiment.get("protocol_version"), "status": status,
            "comparison_group": experiment.get("comparison_group", experiment["name"]),
            "created_at_utc": timestamp, "source_revision": _git_revision(project_root),
            "source_fingerprint": _source_fingerprint(project_root),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "qftlab_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in ("numpy", "scipy", "qiskit", "mlflow")},
            "seeds": {"initialization": run_seed, "data": data_seed},
            "execution": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                          "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                          "tracking": "mlflow" if logger and logger.enabled else "portable_local"},
            "mlflow_run_id": getattr(logger, "run_id", None),
            "tracking_artifacts_uploaded": bool(logger and logger.enabled),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            if logger:
                logger.log_artifacts(run_dir)
        except Exception:
            (run_dir / "tracking_artifact_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            manifest["tracking_artifacts_uploaded"] = False
            manifest["tracking_artifact_error"] = "See tracking_artifact_error.txt"
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if logger:
            logger.end(status)

