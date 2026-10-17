#!/usr/bin/env python3
"""Run a paired zero-offset function-learning sweep over datasets and initializations."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.experiments.function_mse import run_function_mse  # noqa: E402


def parse_ints(value: str, label: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{label} must contain unique integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/function_mse_zero_offset_multidataset.toml")
    parser.add_argument("--data-seeds", default="314159,271828,8675309,20261016,12345",
                        help="Comma-separated independent synthetic dataset seeds.")
    parser.add_argument("--seeds", default="42,43,44,45,46",
                        help="Comma-separated parameter initialization seeds per dataset.")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    data_seeds = parse_ints(args.data_seeds, "--data-seeds")
    init_seeds = parse_ints(args.seeds, "--seeds")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    experiment, data_config = config["experiment"], config["data"]
    if float(data_config["target_offset"]) != 0.0:
        parser.error("This dataset robustness sweep requires target_offset = 0.")

    runs = []
    for data_seed in data_seeds:
        for init_seed in init_seeds:
            run_dir = run_function_mse(config_path, init_seed, project_root=ROOT,
                                       output_root=args.output_root, data_seed=data_seed)
            runs.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))

    runs.sort(key=lambda item: item["test"]["mse"])
    best, worst = runs[0], runs[-1]
    by_dataset: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        by_dataset[run["data_seed"]].append(run)
    dataset_summary = []
    for data_seed in data_seeds:
        group = by_dataset[data_seed]
        mses = [item["test"]["mse"] for item in group]
        fidelities = [item["qft_process_fidelity_diagnostic_only"] for item in group]
        dataset_summary.append({
            "data_seed": data_seed,
            "n_initializations": len(group),
            "test_mse_mean": float(np.mean(mses)),
            "test_mse_population_std": float(np.std(mses)),
            "test_mse_min": float(np.min(mses)),
            "test_mse_max": float(np.max(mses)),
            "dominant_frequency_accuracy": float(np.mean([item["frequency_peak_correct"] for item in group])),
            "qft_process_fidelity_mean": float(np.mean(fidelities)),
            "qft_process_fidelity_population_std": float(np.std(fidelities)),
        })

    values = [item["test"]["mse"] for item in runs]
    fidelities = [item["qft_process_fidelity_diagnostic_only"] for item in runs]
    group_slug = experiment["comparison_group"].replace("-ry-angle-encoding", "")
    result_root = Path(args.output_root or ROOT / config["output"]["root"])
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    group_dir = result_root.parent / group_slug
    group_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "comparison_group": experiment["comparison_group"],
        "protocol_version": experiment["protocol_version"],
        "target": {key: data_config[key] for key in
                   ("target_frequency", "target_offset", "target_cosine", "target_sine")},
        "encoding": "RY(2*pi*x)|0> on q0, x in [0,1), period 1; Pauli-Z readout on q1",
        "training_objective": "training-set MSE only; process fidelity is a diagnostic, not a loss term",
        "dataset_seeds": data_seeds,
        "initialization_seeds_per_dataset": init_seeds,
        "n_datasets": len(data_seeds),
        "n_initializations_per_dataset": len(init_seeds),
        "n_trials": len(runs),
        "dataset_summary": dataset_summary,
        "test_mse_mean_across_trials": float(np.mean(values)),
        "test_mse_population_std_across_trials": float(np.std(values)),
        "test_mse_min": float(best["test"]["mse"]),
        "test_mse_max": float(worst["test"]["mse"]),
        "dominant_frequency_accuracy": float(np.mean([r["frequency_peak_correct"] for r in runs])),
        "qft_process_fidelity_mean": float(np.mean(fidelities)),
        "qft_process_fidelity_population_std": float(np.std(fidelities)),
        "qft_process_fidelity_max": float(np.max(fidelities)),
        "featured_runs": {
            "lowest_test_mse": {key: best[key] for key in ("run_id", "data_seed", "seed", "test",
                                                               "frequency_peak_correct",
                                                               "qft_process_fidelity_diagnostic_only")},
            "highest_test_mse": {key: worst[key] for key in ("run_id", "data_seed", "seed", "test",
                                                                 "frequency_peak_correct",
                                                                 "qft_process_fidelity_diagnostic_only")},
        },
        "trials_sorted_by_test_mse_ascending": runs,
        "scope_note": "All reported target datasets contain exactly one non-DC frequency (k=1), represented by cosine and sine coefficients; these runs do not test multi-frequency recovery.",
    }
    (group_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    colors = plt.get_cmap("tab10")
    fig, (ax, fidelity_ax) = plt.subplots(1, 2, figsize=(13, 5.1))
    for index, dataset in enumerate(data_seeds):
        subset = by_dataset[dataset]
        jitter = np.linspace(-0.10, 0.10, len(subset)) if len(subset) > 1 else np.array([0.0])
        color = colors(index % 10)
        ax.scatter(index + jitter, [r["test"]["mse"] for r in subset], color=color,
                   s=44, alpha=.8, label=f"data seed {dataset}")
        ax.scatter(index, np.mean([r["test"]["mse"] for r in subset]), marker="D",
                   color=color, edgecolor="black", s=58, zorder=4)
        fidelity_ax.scatter([r["qft_process_fidelity_diagnostic_only"] for r in subset],
                            [r["test"]["mse"] for r in subset], color=color, s=44, alpha=.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(data_seeds)), [str(seed) for seed in data_seeds], rotation=25, ha="right")
    ax.set(xlabel="Independent data seed (diamonds: mean across initializations)",
           ylabel="Held-out test MSE", title="Fit across fabricated datasets")
    ax.grid(alpha=.25, which="both")
    fidelity_ax.set_yscale("log")
    fidelity_ax.set(xlabel="QFT process fidelity (diagnostic only)", ylabel="Held-out test MSE",
                    title="Function fit versus QFT recovery")
    fidelity_ax.grid(alpha=.25, which="both")
    fig.suptitle("Zero-offset, single-frequency generalization check")
    fig.tight_layout()
    fig.savefig(group_dir / "dataset_sweep.png", dpi=175)
    plt.close(fig)

    def row(run: dict, rank: str) -> str:
        return (f'<tr><td>{rank}</td><td>{run["data_seed"]}</td><td>{run["seed"]}</td>'
                f'<td>{run["test"]["mse"]:.8g}</td><td>{run["frequency_peak_correct"]}</td>'
                f'<td>{run["qft_process_fidelity_diagnostic_only"]:.6g}</td>'
                f'<td><a href="../runs/{run["run_id"]}/report.html">run report</a></td></tr>')

    featured = [row(best, "Lowest MSE"), row(worst, "Highest MSE")]
    featured_ids = {best["run_id"], worst["run_id"]}
    remainder = [row(run, str(rank)) for rank, run in enumerate(runs, 1)
                 if run["run_id"] not in featured_ids]
    dataset_rows = "\n".join(
        f'<tr><td>{item["data_seed"]}</td><td>{item["n_initializations"]}</td>'
        f'<td>{item["test_mse_mean"]:.8g} ± {item["test_mse_population_std"]:.3g}</td>'
        f'<td>{item["dominant_frequency_accuracy"]:.1%}</td>'
        f'<td>{item["qft_process_fidelity_mean"]:.5g} ± {item["qft_process_fidelity_population_std"]:.3g}</td></tr>'
        for item in dataset_summary)
    html = f'''<!doctype html><meta charset="utf-8"><title>Zero-offset multi-dataset QNN results</title>
<h1>Zero-offset multi-dataset QNN results</h1>
<p>Target: y=0.55 cos(2πx)+0.25 sin(2πx), with zero y-intercept. This is one non-DC frequency, k=1, with cosine and sine quadrature coefficients. Encoding is RY(2πx) on q0; readout is ideal Z on q1. Training uses supervised MSE only. QFT process fidelity is diagnostic only.</p>
<p>{len(data_seeds)} independent dataset seeds × {len(init_seeds)} parameter initializations = {len(runs)} runs. Each dataset has 64/128/512 train/validation/test examples. All frequency scores below are dominant-frequency accuracy; they do not measure multi-frequency recovery.</p>
<h2>Featured extremes</h2>
<p>Lowest test MSE: {best['test']['mse']:.8g} (data seed {best['data_seed']}, initialization {best['seed']}); QFT process fidelity {best['qft_process_fidelity_diagnostic_only']:.6g}.</p>
<p>Highest test MSE: {worst['test']['mse']:.8g} (data seed {worst['data_seed']}, initialization {worst['seed']}); QFT process fidelity {worst['qft_process_fidelity_diagnostic_only']:.6g}.</p>
<img src="dataset_sweep.png" width="96%">
<p><a href="parameter_sensitivity.png">Parameter movement and sensitivity diagnostics</a> · <a href="parameter_sensitivity.json">Sensitivity data</a></p>
<h2>Per-dataset results</h2><table border="1" cellpadding="6"><thead><tr><th>Data seed</th><th>Initializations</th><th>Test MSE mean ± population SD</th><th>k=1 dominant frequency accuracy</th><th>QFT fidelity mean ± population SD</th></tr></thead><tbody>{dataset_rows}</tbody></table>
<h2>Runs (best and worst first)</h2><table border="1" cellpadding="6"><thead><tr><th>Rank</th><th>Data seed</th><th>Init seed</th><th>Test MSE</th><th>k=1 peak correct</th><th>QFT process fidelity</th><th>Report</th></tr></thead><tbody>{' '.join(featured + remainder)}</tbody></table>
<p><a href="../../HYPOTHESIS_TREE.md">Research hypothesis tree</a> · <a href="summary.json">Machine-readable results</a></p>'''
    (group_dir / "index.html").write_text(html, encoding="utf-8")
    print(group_dir / "index.html")
    print(json.dumps({key: summary[key] for key in
                      ("n_datasets", "n_trials", "test_mse_mean_across_trials",
                       "test_mse_min", "test_mse_max", "dominant_frequency_accuracy",
                       "qft_process_fidelity_mean")}, indent=2))


if __name__ == "__main__":
    main()
