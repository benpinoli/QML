#!/usr/bin/env python3
"""Regenerate saved scalar-MSE run plots and reports without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qftlab.experiments.function_mse import refresh_function_mse_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path,
                        default=ROOT / "results/periodic-function-mse-z1-readout")
    args = parser.parse_args()
    group = args.group if args.group.is_absolute() else ROOT / args.group
    summary = json.loads((group / "summary.json").read_text(encoding="utf-8"))
    for run_id in summary["run_ids"]:
        run_dir = ROOT / "results/runs" / run_id
        refresh_function_mse_run(run_dir)
        print(run_dir / "report.html")


if __name__ == "__main__":
    main()
