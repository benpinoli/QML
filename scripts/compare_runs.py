#!/usr/bin/env python3
"""Build an HTML/CSV summary from portable result bundles."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/comparison.html"))
    parser.add_argument("--comparison-group", help="select one declared comparison group")
    args = parser.parse_args()
    rows = []
    run_paths = []
    for summary_path in sorted(args.runs.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = summary_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows.append({**summary, "protocol_version": manifest.get("protocol_version", ""),
                     "comparison_group": manifest.get("comparison_group", summary.get("comparison_group", "")),
                     "source_revision": manifest.get("source_revision", ""),
                     "report": str(summary_path.parent / "artifacts/report.html")})
        run_paths.append(summary_path.parent)
    if args.comparison_group:
        selected = [(row, path) for row, path in zip(rows, run_paths) if row.get("comparison_group") == args.comparison_group]
        rows = [item[0] for item in selected]
        run_paths = [item[1] for item in selected]
    elif len({row.get("comparison_group") for row in rows}) > 1:
        parser.error("Found multiple comparison groups. Select one with --comparison-group.")
    compatibility_fields = ("protocol_version", "comparison_group", "data_seed", "initialization_scale",
                            "max_iterations", "train_size", "validation_size", "test_size", "ansatz")
    mismatched = [field for field in compatibility_fields if len({str(row.get(field)) for row in rows}) > 1]
    if mismatched:
        parser.error("Runs differ on controlled comparison fields: " + ", ".join(mismatched))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run_id", "status", "optimizer", "seed", "protocol_version", "comparison_group", "initial_process_fidelity",
               "qft_process_fidelity", "test_state_fidelity_mean", "test_output_total_variation_mean",
               "frequency_accuracy", "frequency_cross_entropy", "iterations", "report"]
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>" for row in rows)
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    args.output.write_text(
        "<!doctype html><meta charset='utf-8'><title>QFT recovery runs</title>"
        "<style>body{font:14px system-ui;margin:2rem;color:#202733}table{border-collapse:collapse}"
        "td,th{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}tr:hover{background:#f3f6fa}</style>"
        f"<h1>QFT recovery runs</h1><p>{len(rows)} portable run bundles found.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{table_rows}</tbody></table>", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for run_dir, row in zip(run_paths, rows):
            points = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
            ax.plot([point["step"] for point in points], [point["process_fidelity"] for point in points],
                    alpha=.75, label=f"seed {row.get('seed')} / {row.get('optimizer')}")
        ax.set(xlabel="Optimizer iteration", ylabel="Process fidelity", ylim=(0, 1.02),
               title="QFT recovery trajectories across runs")
        ax.grid(alpha=.25)
        if len(rows) <= 12:
            ax.legend(fontsize=8)
        fig.tight_layout()
        convergence_path = args.output.parent / "comparison_convergence.png"
        fig.savefig(convergence_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        score_values = [float(row["test_state_fidelity_mean"]) for row in rows if row.get("test_state_fidelity_mean") is not None]
        labels = [f"{row.get('optimizer')}\nseed {row.get('seed')}" for row in rows if row.get("test_state_fidelity_mean") is not None]
        ax.bar(range(len(score_values)), score_values, color="#4389ad")
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Mean held-out state fidelity",
               ylim=(0, 1.05), title="Held-out performance per run")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        score_path = args.output.parent / "comparison_test_scores.png"
        fig.savefig(score_path, dpi=160)
        plt.close(fig)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write("<h2>Visual comparisons</h2><img style='max-width:100%' src='comparison_convergence.png'>")
            stream.write("<img style='max-width:100%' src='comparison_test_scores.png'>")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
