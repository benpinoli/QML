#!/usr/bin/env python3
"""Recompute parameter movement and local sensitivity diagnostics for a run group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.circuits.qft import PARAMETER_NAMES  # noqa: E402
from qftlab.experiments.function_mse import predict  # noqa: E402


def main() -> None:
    group_dir = ROOT / "results/periodic-function-mse-zero-offset-multidataset-z1-readout"
    group = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    h = 1e-5
    grid = np.arange(512, dtype=float) / 512
    records = []
    for run in group["trials_sorted_by_test_mse_ascending"]:
        run_dir = ROOT / "results/runs" / run["run_id"]
        params = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
        with np.load(run_dir / "dataset.npz") as dataset:
            x, y = dataset["train_x"], dataset["train_y"]
        initial = np.array([params["initial"][name] for name in PARAMETER_NAMES])
        final = np.array([params["final"][name] for name in PARAMETER_NAMES])

        def loss(theta: np.ndarray) -> float:
            return float(np.mean((predict(theta, x) - y) ** 2))

        def central_gradient(theta: np.ndarray) -> np.ndarray:
            result = np.empty(len(theta))
            for index in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[index] = h
                result[index] = (loss(theta + delta) - loss(theta - delta)) / (2 * h)
            return result

        prediction_jacobian_rms = []
        for index in range(len(final)):
            delta = np.zeros_like(final)
            delta[index] = h
            derivative = (predict(final + delta, grid) - predict(final - delta, grid)) / (2 * h)
            prediction_jacobian_rms.append(float(np.sqrt(np.mean(derivative**2))))
        wrapped_change = (final - initial + np.pi) % (2 * np.pi) - np.pi
        records.append({
            "run_id": run["run_id"], "data_seed": run["data_seed"], "initialization_seed": run["seed"],
            "largest_wrapped_change_parameter": PARAMETER_NAMES[int(np.argmax(np.abs(wrapped_change)))],
            "wrapped_angle_change_radians": dict(zip(PARAMETER_NAMES, map(float, wrapped_change))),
            "initial_training_loss_gradient_central_difference": dict(
                zip(PARAMETER_NAMES, map(float, central_gradient(initial)))),
            "final_training_loss_gradient_central_difference": dict(
                zip(PARAMETER_NAMES, map(float, central_gradient(final)))),
            "final_prediction_jacobian_rms_on_512_point_grid": dict(
                zip(PARAMETER_NAMES, prediction_jacobian_rms)),
        })

    aggregate = {}
    for name in PARAMETER_NAMES:
        changes = [abs(row["wrapped_angle_change_radians"][name]) for row in records]
        initial_grad = [abs(row["initial_training_loss_gradient_central_difference"][name]) for row in records]
        final_grad = [abs(row["final_training_loss_gradient_central_difference"][name]) for row in records]
        jacobian = [row["final_prediction_jacobian_rms_on_512_point_grid"][name] for row in records]
        aggregate[name] = {
            "mean_absolute_wrapped_change_radians": float(np.mean(changes)),
            "largest_change_count": sum(row["largest_wrapped_change_parameter"] == name for row in records),
            "median_absolute_initial_loss_derivative": float(np.median(initial_grad)),
            "median_absolute_final_loss_derivative": float(np.median(final_grad)),
            "median_final_prediction_jacobian_rms": float(np.median(jacobian)),
        }
    output = {
        "comparison_group": group["comparison_group"],
        "n_runs": len(records),
        "finite_difference_method": "Central difference; h=1e-5 radians per parameter.",
        "prediction_jacobian_grid": "512 equally spaced x values over [0,1).",
        "warning": "This is a post-hoc diagnostic, not the exact gradient history used by L-BFGS-B. Parameter movement is an optimizer step and is not itself a sensitivity measure.",
        "aggregate_by_parameter": aggregate,
        "runs": records,
    }
    output_path = group_dir / "parameter_sensitivity.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    labels = list(PARAMETER_NAMES)
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].bar(positions, [aggregate[name]["mean_absolute_wrapped_change_radians"] for name in labels], color="slateblue")
    axes[0].set_ylabel("Mean |wrapped Δ| (rad)")
    axes[1].bar(positions, [max(aggregate[name]["median_absolute_initial_loss_derivative"], 1e-16)
                            for name in labels], color="darkorange")
    axes[1].set_yscale("log"); axes[1].set_ylabel("Median |dMSE/dθ| at init")
    axes[2].bar(positions, [max(aggregate[name]["median_final_prediction_jacobian_rms"], 1e-16)
                            for name in labels], color="forestgreen")
    axes[2].set_yscale("log"); axes[2].set_ylabel("Median prediction Jacobian RMS")
    axes[2].set_xticks(positions, labels, rotation=25, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=.25, which="both")
    fig.suptitle("Parameter movement and sensitivity (25 zero-offset runs)")
    fig.tight_layout()
    fig.savefig(group_dir / "parameter_sensitivity.png", dpi=175)
    plt.close(fig)
    print(output_path)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
