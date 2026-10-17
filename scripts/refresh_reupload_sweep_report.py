#!/usr/bin/env python3
"""Rebuild aggregate plots/table from saved re-upload run summaries, without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reupload_sweep import GROUP_DIR, _write_group_outputs  # noqa: E402


def main() -> None:
    summary_path = GROUP_DIR / "summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = saved["runs_sorted_by_test_mse"]
    updated = _write_group_outputs(
        runs,
        saved["uploads_tested"],
        len(saved["initialization_seeds"]),
        saved["data_seed"],
        saved["random_circuit_spectral_reachability"],
        saved["one_upload_baseline_comparison"],
    )
    print(json.dumps({"report": str(GROUP_DIR / "index.html"), "run_count": updated["n_runs"]}, indent=2))


if __name__ == "__main__":
    main()
