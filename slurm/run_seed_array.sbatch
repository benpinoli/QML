#!/usr/bin/env bash
# Each array task runs one initialization seed against the same fixed data split.
# Adjust the array range and resource requests to local policy.
#SBATCH --job-name=qft-seeds
#SBATCH --array=0-9%4
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
PYTHON_BIN="${QFTLAB_PYTHON:-python}"
SEED="$((4200 + SLURM_ARRAY_TASK_ID))"
"${PYTHON_BIN}" scripts/run_experiment.py --config configs/qft_recovery.toml --seed "${SEED}" --output-root "${QFTLAB_OUTPUT_ROOT:-results/runs}"
