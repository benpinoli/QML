#!/usr/bin/env python3
"""Run seeded supervised MSE trials for period-one scalar function learning."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.experiments.function_mse import run_function_mse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/function_mse.toml")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                        help="Comma-separated initialization seeds (data split is fixed).")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Provide one or more unique integer seeds.")
    runs = [run_function_mse(args.config, seed, project_root=ROOT, output_root=args.output_root)
            for seed in seeds]
    summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in runs]
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    experiment = config["experiment"]
    data_config = config["data"]
    result_root = Path(args.output_root or ROOT / "results/runs")
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    group_slug = experiment["comparison_group"].replace("-ry-angle-encoding", "")
    group_dir = result_root.parent / group_slug
    group_dir.mkdir(parents=True, exist_ok=True)
    values = [item["test"]["mse"] for item in summaries]
    payload = {
        "comparison_group": experiment["comparison_group"],
        "training_objective": "training-set MSE only; no process/Frobenius alignment or PINN residual",
        "encoding": "RY(2*pi*x)|0> on q0, x in [0,1), period 1; Pauli-Z readout on q1",
        "data_seed": int(data_config["seed"]),
        "target": {key: data_config[key] for key in ("target_frequency", "target_offset", "target_cosine", "target_sine")},
        "paired_comparison_group": experiment.get("paired_comparison_group"),
        "run_ids": [item["run_id"] for item in summaries],
        "seeds": seeds,
        "n_trials": len(summaries),
        "test_mse_mean": sum(values) / len(values),
        "test_mse_std_population": (sum((value - sum(values)/len(values))**2 for value in values) / len(values))**0.5,
        "mean_iterations": sum(item["iterations"] for item in summaries) / len(summaries),
        "frequency_peak_accuracy": sum(item["frequency_peak_correct"] for item in summaries) / len(summaries),
        "trials": summaries,
    }
    (group_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    seeds_axis = [item["seed"] for item in summaries]
    constant = summaries[0]["constant_baseline_test"]["mse"]
    oracle = summaries[0]["known_frequency_fourier_baseline_test"]["mse"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.scatter(seeds_axis, values, s=52, label="QNN test MSE", zorder=3)
    ax.axhline(constant, color="C1", linestyle="--", label="constant baseline")
    ax.axhline(oracle, color="C2", linestyle=":", label="Fourier regression (known k=1)")
    ax.set(xlabel="Parameter initialization seed", ylabel="Held-out test MSE",
           title=f"{experiment['name']}: MSE-only QNN, RY(2πx), Z₁")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    if max(values) < 0.005:
        inset = ax.inset_axes([0.08, 0.48, 0.36, 0.34])
        inset.scatter(seeds_axis, values, s=24, color="C0", zorder=3)
        inset.axhline(oracle, color="C2", linestyle=":")
        inset.set_ylim(0, max(values) * 1.35)
        inset.set_title("Near-zero MSE detail", fontsize=9)
        inset.tick_params(labelsize=8)
        inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        inset.grid(alpha=.2)
    fig.savefig(group_dir / "trial_test_mse.png", dpi=170); plt.close(fig)
    links = "\n".join(f'<tr><td>{s["seed"]}</td><td>{s["iterations"]}</td><td>{s["test"]["mse"]:.8g}</td>'
                       f'<td>{s["validation_mse"]:.8g}</td><td>{s["frequency_peak_correct"]}</td>'
                       f'<td><a href="../runs/{s["run_id"]}/report.html">run report</a></td></tr>'
                       for s in summaries)
    html = f'''<!doctype html><meta charset="utf-8"><title>{experiment['name']} trial group</title>
<h1>{experiment['name']} trial group</h1>
<p>Objective: supervised MSE on (x,y) pairs only. No Frobenius/process-fidelity alignment or PINN residual is used in training.<br>
Encoding: RY(2πx)|0⟩ on qubit 0, x∈[0,1), period 1. Readout: ideal Pauli-Z expectation on qubit 1.</p>
<p>Target: y={data_config['target_offset']}+{data_config['target_cosine']} cos(2πx)+{data_config['target_sine']} sin(2πx). Data seed: {data_config['seed']}; input split sizes: {data_config['train_size']}/{data_config['validation_size']}/{data_config['test_size']}.</p>
<p>Seeds: {seeds}. Mean test MSE: {payload['test_mse_mean']:.8g}; population SD: {payload['test_mse_std_population']:.8g}; mean iterations: {payload['mean_iterations']:.2f}; peak-frequency accuracy: {payload['frequency_peak_accuracy']:.3f}.</p>
{f'<p>Paired comparison group: <a href="../{experiment["paired_comparison_group"].replace("-ry-angle-encoding", "")}/index.html">{experiment["paired_comparison_group"]}</a>. Only the target offset differs; input points and split are fixed by the shared data seed.</p>' if experiment.get('paired_comparison_group') else ''}
<img src="trial_test_mse.png" width="75%">
<p><a href="../../HYPOTHESIS_TREE.md">Research hypothesis tree</a></p>
<table border="1" cellpadding="6"><thead><tr><th>Seed</th><th>Iterations</th><th>Test MSE</th><th>Validation MSE</th><th>Peak frequency correct</th><th>Run</th></tr></thead><tbody>{links}</tbody></table>'''
    (group_dir / "index.html").write_text(html, encoding="utf-8")
    print(group_dir / "index.html")
    print(json.dumps({k: payload[k] for k in ("n_trials", "test_mse_mean", "test_mse_std_population", "mean_iterations", "frequency_peak_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
