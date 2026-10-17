#!/usr/bin/env python3
"""Download MLflow artifacts for one run into a portable local bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    try:
        import mlflow
        from mlflow.artifacts import download_artifacts
    except ImportError as exc:
        parser.error("MLflow export requires `pip install -e '.[tracking]'`.")
    run = mlflow.get_run(args.run_id)
    destination = args.output / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    download_artifacts(run_id=args.run_id, dst_path=str(destination))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
