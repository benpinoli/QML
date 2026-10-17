#!/usr/bin/env python3
"""Regenerate component-sweep aggregate plots from saved summary.json only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.visualization.parameter_summaries import (  # noqa: E402
    categorical_run_colors,
    generate_parameter_summary_plots,
    refresh_group_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, default=Path(
        "results/periodic-function-mse-zero-offset-component-sweep-z1-readout/summary.json"))
    args = parser.parse_args()
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary["runs_sorted_by_test_mse"]
    color_by_run = categorical_run_colors(runs)
    summary["run_color_by_id"] = color_by_run
    for run in runs:
        run["plot_color"] = color_by_run[run["run_id"]]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    metrics_path = summary_path.parent / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metric_rows = list(csv.DictReader(stream))
            columns = list(metric_rows[0]) if metric_rows else []
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in metric_rows:
                row["plot_color"] = color_by_run[row["run_id"]]
                writer.writerow(row)

    outputs = generate_parameter_summary_plots(runs, color_by_run, summary_path.parent)
    refresh_group_report(summary_path.parent)
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "plots_refreshed": outputs,
                      "manifest": str((summary_path.parent / "parameter_plot_manifest.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
