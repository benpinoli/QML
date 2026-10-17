#!/usr/bin/env python3
"""Run nested zero-intercept multi-frequency targets and create group visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.circuits.qft import PARAMETER_NAMES  # noqa: E402
from qftlab.experiments.function_mse import run_function_mse  # noqa: E402
from qftlab.visualization.parameter_summaries import (  # noqa: E402
    categorical_run_colors,
    generate_parameter_summary_plots,
)


CONFIGS = {
    2: ROOT / "configs/function_mse_components_2.toml",
    3: ROOT / "configs/function_mse_components_3.toml",
    4: ROOT / "configs/function_mse_components_4.toml",
}
GROUP_DIR = ROOT / "results/periodic-function-mse-zero-offset-component-sweep-z1-readout"


def wrapped(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def target_support(components: list[dict]) -> list[int]:
    return [int(item["frequency"]) for item in components]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                        help="Ten or more comma-separated parameter initialization seeds.")
    parser.add_argument("--data-seed", type=int, default=314159,
                        help="Shared fabricated-data seed, keeping the x split paired across targets.")
    args = parser.parse_args()
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain unique integers")

    runs: list[dict] = []
    for n_components, config_path in CONFIGS.items():
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        if float(config["data"]["target_offset"]) != 0.0:
            parser.error(f"{config_path} must have a zero intercept")
        for seed in seeds:
            run_dir = run_function_mse(config_path, seed, project_root=ROOT, data_seed=args.data_seed)
            run = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            run["n_components"] = n_components
            run["run_dir"] = str(run_dir.relative_to(ROOT))
            parameters = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
            run["initial_parameters"] = parameters["initial"]
            run["final_parameters"] = parameters["final"]
            with np.load(run_dir / "evaluation_outputs.npz") as outputs:
                predicted = outputs["predicted_spectrum"]
                target = outputs["target_spectrum"]
                maximum_frequency = min(16, len(target) - 1)
                bins = np.arange(1, maximum_frequency + 1)
                pred_amplitudes = 2 * np.abs(predicted[bins])
                target_amplitudes = 2 * np.abs(target[bins])
                threshold = 1e-3
                predicted_support = bins[pred_amplitudes > threshold].astype(int).tolist()
                true_support = target_support(run["target_components"])
                true_positive = len(set(predicted_support) & set(true_support))
                run["frequency_support"] = {
                    "threshold_function_amplitude": threshold,
                    "target_non_dc_bins": true_support,
                    "predicted_non_dc_bins": predicted_support,
                    "precision": true_positive / len(predicted_support) if predicted_support else 0.0,
                    "recall": true_positive / len(true_support) if true_support else 1.0,
                    "f1": (2 * true_positive / (len(predicted_support) + len(true_support))
                           if predicted_support or true_support else 1.0),
                    "target_amplitude_by_frequency": {
                        str(int(k)): float(target_amplitudes[index]) for index, k in enumerate(bins)
                    },
                    "predicted_amplitude_by_frequency": {
                        str(int(k)): float(pred_amplitudes[index]) for index, k in enumerate(bins)
                    },
                }
            run["wrapped_angle_change_radians"] = {
                name: wrapped(run["final_parameters"][name] - run["initial_parameters"][name])
                for name in PARAMETER_NAMES
            }
            runs.append(run)

    runs.sort(key=lambda row: row["test"]["mse"])
    best, worst = runs[0], runs[-1]
    component_groups = []
    for n_components in CONFIGS:
        subset = [run for run in runs if run["n_components"] == n_components]
        config = tomllib.loads(CONFIGS[n_components].read_text())
        mse_values = [item["test"]["mse"] for item in subset]
        fid = [item["qft_process_fidelity_diagnostic_only"] for item in subset]
        component_groups.append({
            "n_components": n_components,
            "target_offset": float(config["data"]["target_offset"]),
            "target_components": config["data"]["target_components"],
            "test_mse_mean": float(np.mean(mse_values)),
            "test_mse_population_std": float(np.std(mse_values)),
            "dominant_frequency_accuracy": float(np.mean([item["frequency_peak_correct"] for item in subset])),
            "frequency_support_precision_mean": float(np.mean([item["frequency_support"]["precision"] for item in subset])),
            "frequency_support_recall_mean": float(np.mean([item["frequency_support"]["recall"] for item in subset])),
            "frequency_support_f1_mean": float(np.mean([item["frequency_support"]["f1"] for item in subset])),
            "qft_process_fidelity_mean": float(np.mean(fid)),
            "qft_process_fidelity_population_std": float(np.std(fid)),
            "qft_process_fidelity_max": float(np.max(fid)),
            "best_run_id": min(subset, key=lambda item: item["test"]["mse"])["run_id"],
            "worst_run_id": max(subset, key=lambda item: item["test"]["mse"])["run_id"],
        })

    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    color_by_run = categorical_run_colors(runs)
    for run in runs:
        run["plot_color"] = color_by_run[run["run_id"]]

    summary = {
        "comparison_group": "periodic-function-mse-zero-offset-component-sweep-z1-readout",
        "objective": "Training-set MSE only; process fidelity is diagnostic only.",
        "encoding": "RY(2*pi*x)|0> on q0, x in [0,1), period 1; ideal Pauli-Z readout on q1",
        "target_family": "g_N(x)=sum_{k=1}^N [a_k cos(2*pi*k*x)+b_k sin(2*pi*k*x)] with zero intercept",
        "shared_data_seed": args.data_seed,
        "dataset_sizes": {"train": 64, "validation": 128, "test": 512},
        "initialization_seeds": seeds,
        "n_initializations_per_component_count": len(seeds),
        "n_runs": len(runs),
        "components": component_groups,
        "frequency_support_metric": "Non-DC spectral amplitude threshold 1e-3 on the dense 2048-point grid; precision/recall/F1 are reported per run and averaged per target.",
        "featured_runs": {
            "lowest_test_mse": {"run_id": best["run_id"], "n_components": best["n_components"],
                                 "seed": best["seed"], "test_mse": best["test"]["mse"]},
            "highest_test_mse": {"run_id": worst["run_id"], "n_components": worst["n_components"],
                                  "seed": worst["seed"], "test_mse": worst["test"]["mse"]},
        },
        "run_color_by_id": color_by_run,
        "runs_sorted_by_test_mse": runs,
    }
    (GROUP_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    generate_parameter_summary_plots(runs, color_by_run, GROUP_DIR)

    csv_path = GROUP_DIR / "metrics.csv"
    columns = ["run_id", "n_components", "data_seed", "initialization_seed", "n_train", "n_validation",
               "n_test", "train_mse", "validation_mse", "test_mse", "dominant_frequency_correct",
               "target_support", "predicted_support", "support_precision", "support_recall", "support_f1",
               "qft_process_fidelity", "plot_color"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            sizes = run["dataset_sizes"]
            writer.writerow({
                "run_id": run["run_id"], "n_components": run["n_components"], "data_seed": run["data_seed"],
                "initialization_seed": run["seed"], "n_train": sizes["train"],
                "n_validation": sizes["validation"], "n_test": sizes["test"],
                "train_mse": run["final_train_mse"], "validation_mse": run["validation_mse"],
                "test_mse": run["test"]["mse"],
                "dominant_frequency_correct": run["frequency_peak_correct"],
                "target_support": ";".join(map(str, run["frequency_support"]["target_non_dc_bins"])),
                "predicted_support": ";".join(map(str, run["frequency_support"]["predicted_non_dc_bins"])),
                "support_precision": run["frequency_support"]["precision"],
                "support_recall": run["frequency_support"]["recall"], "support_f1": run["frequency_support"]["f1"],
                "qft_process_fidelity": run["qft_process_fidelity_diagnostic_only"], "plot_color": run["plot_color"],
            })

    # Lowest and highest MSE runs are deliberately the first two table rows.
    featured_ids = {best["run_id"], worst["run_id"]}
    display_runs = [best, worst] + [run for run in runs if run["run_id"] not in featured_ids]
    rows = "\n".join(
        f'<tr><td>{"Lowest MSE" if run["run_id"] == best["run_id"] else "Highest MSE" if run["run_id"] == worst["run_id"] else ""}</td>'
        f'<td>{run["n_components"]}</td><td>{run["data_seed"]}</td><td>{run["seed"]}</td>'
        f'<td>{run["dataset_sizes"]["train"]}</td><td>{run["test"]["mse"]:.7g}</td>'
        f'<td>{run["frequency_support"]["target_non_dc_bins"]}</td>'
        f'<td>{run["frequency_support"]["predicted_non_dc_bins"]}</td>'
        f'<td>{run["frequency_support"]["recall"]:.2f}</td>'
        f'<td>{run["qft_process_fidelity_diagnostic_only"]:.5g}</td>'
        f'<td><a href="../runs/{run["run_id"]}/report.html">report</a></td></tr>'
        for run in display_runs)
    group_rows = "\n".join(
        f'<tr><td>{item["n_components"]}</td><td>{item["target_components"]}</td>'
        f'<td>{item["test_mse_mean"]:.6g} ± {item["test_mse_population_std"]:.2g}</td>'
        f'<td>{item["dominant_frequency_accuracy"]:.0%}</td>'
        f'<td>{item["frequency_support_precision_mean"]:.2f}</td>'
        f'<td>{item["frequency_support_recall_mean"]:.2f}</td>'
        f'<td>{item["qft_process_fidelity_mean"]:.4f} ± {item["qft_process_fidelity_population_std"]:.2g}</td></tr>'
        for item in component_groups)
    circular_links = " ".join(
        f'<details><summary>{name}</summary><img src="{name}_circular_changes.png" width="95%"></details>'
        for name in PARAMETER_NAMES)
    html = f'''<!doctype html><meta charset="utf-8"><title>Zero-intercept multi-frequency QNN sweep</title>
<h1>Zero-intercept multi-frequency QNN sweep</h1>
<p>Nested targets have 2, 3, or 4 distinct non-DC frequencies (k=1..N), with a zero y-intercept. Every added target retains the same coefficients for existing components. MSE is the training objective; process fidelity is diagnostic only.</p>
<p><b>Data volume per run:</b> {summary['dataset_sizes']['train']} train + {summary['dataset_sizes']['validation']} validation + {summary['dataset_sizes']['test']} test points. Shared data seed {args.data_seed} keeps x samples paired across target complexities. Ten initialization trials per target; {len(runs)} runs total.</p>
<h2>Aggregate by number of frequency components</h2><table border="1" cellpadding="6"><thead><tr><th>Components</th><th>Target (k, cosine, sine)</th><th>Test MSE mean ± SD</th><th>Dominant k=1 accuracy</th><th>Support precision</th><th>Support recall</th><th>QFT process fidelity mean ± SD</th></tr></thead><tbody>{group_rows}</tbody></table>
<h2>Angle movement histograms</h2><p>Each histogram uses the shortest signed angular change modulo 2π. Side-by-side bars retain blue (2), orange (3), and green (4 frequency) counts even when they fall in the same bin.</p><img src="parameter_change_histograms.png" width="96%">
<h2>All seven gates by run</h2><p>Each spoke is one angle; radius maps the wrapped position from −π at center to +π at the outer circle. Each run uses one color: initial polygon dashed with circle markers, final polygon solid with square markers. Thin radial connectors join each gate's initial and final positions. Colors are distinct within each frequency-count group and reused across groups by initialization seed.</p><img src="parameter_spider_profiles.png" width="96%">
<h2>Individual gate circles</h2><p>Each colored arc connects the initial angle marker (hollow circle) to the final marker (filled square) along the shortest circular path. All connectors are drawn first, then all markers are drawn above them with contrasting outlines. An unchanged angle has overlapping endpoints and no visible arc. Colors are distinct within each frequency-count group and reused across groups by initialization seed; run IDs and hex colors are in summary.json and metrics.csv.</p>{circular_links}
<h2>Runs, with global MSE extremes first</h2><table border="1" cellpadding="6"><thead><tr><th>Featured</th><th>Components</th><th>Data seed</th><th>Init seed</th><th>Train points</th><th>Test MSE</th><th>Target support</th><th>Learned support</th><th>Support recall</th><th>QFT fidelity</th><th>Run</th></tr></thead><tbody>{rows}</tbody></table>
<p><a href="metrics.csv">Download run metrics CSV</a> · <a href="summary.json">Machine-readable summary</a> · <a href="parameter_plot_manifest.json">Plot methods and manifest</a> · <a href="../../HYPOTHESIS_TREE.md">Hypothesis tree</a></p>
<p>Regenerate only these plots from saved results: <code>python scripts/refresh_component_sweep_visualizations.py</code>. The plotting implementation is in <code>src/qftlab/visualization/parameter_summaries.py</code>; no retraining is performed.</p>'''
    (GROUP_DIR / "index.html").write_text(html, encoding="utf-8")
    print(GROUP_DIR / "index.html")
    print(json.dumps({"runs": len(runs), "components": component_groups,
                      "lowest_mse": summary["featured_runs"]["lowest_test_mse"],
                      "highest_mse": summary["featured_runs"]["highest_test_mse"]}, indent=2))


if __name__ == "__main__":
    main()
