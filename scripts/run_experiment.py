#!/usr/bin/env python3
"""Run one configured experiment from a source checkout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qftlab.experiments.qft_recovery import run_qft_recovery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qft_recovery.toml")
    parser.add_argument("--seed", type=int, help="override the initialization seed; data split seed stays fixed")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root
    if output_root is None and os.environ.get("QFTLAB_OUTPUT_ROOT"):
        output_root = Path(os.environ["QFTLAB_OUTPUT_ROOT"])
    path = run_qft_recovery(args.config, seed=args.seed, output_root=output_root, project_root=PROJECT_ROOT)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
