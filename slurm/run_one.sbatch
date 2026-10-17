#!/usr/bin/env bash
# Adjust account, partition, time, and resources to local supercomputer policy.
# Submit from the repository root: sbatch slurm/run_one.sbatch
# Optional: set QFTLAB_PYTHON to the environment's Python executable.
#SBATCH --job-name=qft-recovery
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
PYTHON_BIN="${QFTLAB_PYTHON:-python}"
"${PYTHON_BIN}" scripts/run_experiment.py --config configs/qft_recovery.toml --output-root "${QFTLAB_OUTPUT_ROOT:-results/runs}"
