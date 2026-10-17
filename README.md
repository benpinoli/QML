# QFT-Initialized Quantum Neural Network Research Lab

This repository is a research workspace for studying how well quantum neural networks (QNNs) can learn functions, with a first controlled experiment: can a randomized, trainable two-qubit circuit learn the known two-qubit quantum Fourier transform (QFT)?

The first goal is deliberately narrow. Establish a reproducible training loop, a trustworthy objective, fabricated datasets, and useful diagnostics. Once the circuit can recover the QFT reliably, the same tools can be used to study function learning, frequency recovery, and alternative optimizers.

## Research questions

- Can gradient-based training recover a known quantum algorithm from randomized parameters when the circuit has the QFT ansatz structure?
- How do initialization, data coverage, shot noise, and optimizer choice affect convergence and repeatability?
- When the model is trained on input/output examples rather than a full unitary target, does it recover the QFT itself or only a circuit that agrees on the training set?
- How accurately can a trained QNN represent target functions with known Fourier spectra, including spectra with multiple frequencies?

## First benchmark: recover the two-qubit QFT

The first implemented circuit uses the canonical two-qubit QFT as a reference and an ansatz with the same `U – controlled phase – U – SWAP` layout. Its seven parameters are randomized around the canonical angles. The circuit convention documents Qiskit wire ordering, positive controlled-phase sign, final swap, and global-phase-insensitive scoring.

The baseline uses an ideal unitary-level objective and maximizes phase-insensitive process fidelity

`F(U, V) = |Tr(U† V)|² / d²`, where `d = 4`,

by minimizing `1 - F`. This is a QFT calibration experiment: the target unitary is supplied to the optimizer. Seeded state examples are generated for separate train/validation/test diagnostics, and the held-out test split is evaluated only after optimization. The next protocol should train from example input/output pairs instead of the full unitary.

### Frequency benchmark

QFT frequency behavior should be evaluated with states whose amplitudes encode known periodic structure or phase ramps. Generate a controlled set of such inputs with known frequency labels, apply the target QFT, and measure whether the output distribution peaks at the expected bin. Report top-1 frequency accuracy, frequency-bin cross-entropy, and the distance between predicted and ideal output distributions. Include uniform/random controls so an uninformative output cannot score well by chance.

Keep this frequency task separate from unitary recovery: a circuit can classify selected frequencies correctly without implementing the full QFT. Together, the two views show whether the learned circuit is algorithmically close and whether it performs the intended frequency operation.

For the forthcoming one-dimensional periodic function-regression protocol, the chosen classical feature map is `x ∈ [0,1) → RY(2πx)|0⟩`, followed by the QFT-layout variational block and a Pauli-Z expectation readout. This choice is documented with its domain and reversibility limits in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). The current QFT-recovery run does not use this classical feature encoding.

The formal LaTeX specification for the supervised function-learning experiment—including the data protocol, MSE objective, spectral evaluation, baselines, run artifacts, and first trial results—is [METHODOLOGY.tex](METHODOLOGY.tex).

Run ten initialization trials on the fixed synthetic periodic-function split with:

```bash
python scripts/run_function_mse.py --config configs/function_mse.toml
```

Each seed writes a portable run bundle; the grouped report and comparison plot are written under `results/periodic-function-mse-z1-readout/`. This is a separate experiment from QFT unitary recovery.

The paired zero-offset proof of concept uses the same inputs and five initialization seeds (42–46):

```bash
python scripts/run_function_mse.py --config configs/function_mse_zero_offset.toml --seeds 42,43,44,45,46
```

Its report is at [`results/periodic-function-mse-zero-offset-z1-readout/index.html`](results/periodic-function-mse-zero-offset-z1-readout/index.html). It achieved test MSE at numerical precision on the zero-offset target; this is five initializations on one fixed dataset, not a multi-dataset robustness result.

To reproduce the five-dataset, five-initialization sweep (all targets retain zero intercept and the same single (k=1) frequency):

```bash
python scripts/run_function_mse_dataset_sweep.py
```

The grouped report is [`results/periodic-function-mse-zero-offset-multidataset-z1-readout/index.html`](results/periodic-function-mse-zero-offset-multidataset-z1-readout/index.html). It puts the lowest- and highest-test-MSE runs first and links each per-run report. Override the comma-separated `--data-seeds` and `--seeds` options for another sweep.

For the zero-intercept nested targets with two, three, and four distinct frequencies (10 initializations each):

```bash
python scripts/run_function_component_sweep.py
```

The summary graphs, run-level metrics CSV (including training/validation/test counts), and run links are at [`results/periodic-function-mse-zero-offset-component-sweep-z1-readout/index.html`](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/index.html). Per-run bundles remain under `results/runs/`.

Aggregate parameter plots are generated by the reusable module `src/qftlab/visualization/parameter_summaries.py`; the group report links a `parameter_plot_manifest.json` that records the plotting definitions. Regenerate the plots from the saved `summary.json` without rerunning training:

```bash
python scripts/refresh_component_sweep_visualizations.py
```

The two-qubit data re-uploading experiment compares one through four uploads on the same zero-intercept targets, data split, and initialization seeds. Its grouped report is [`results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/index.html`](results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/index.html). Run it with `python scripts/run_reupload_sweep.py`; regenerate its summary plots and tables from saved run summaries with `python scripts/refresh_reupload_sweep_report.py`.

Research questions, candidate explanations, evidence, and planned discriminating experiments are maintained as a [human-readable hypothesis tree](HYPOTHESIS_TREE.md) backed by [machine-readable JSON](hypotheses/HYPOTHESIS_TREE.json) and its [schema](hypotheses/HYPOTHESIS_TREE.schema.json).

## Data and experiment design

- **Fabricated, deterministic datasets:** generate fixed train, validation, and test splits from seeded inputs. Store the generator configuration and seed with every run rather than committing opaque samples.
- **QFT recovery examples:** sample computational basis states and selected superpositions; derive target statevectors or measurement distributions from the reference QFT.
- **Known-frequency examples:** generate periodic sequences or phase-ramp states with known frequency bins, then produce ideal QFT labels/distributions.
- **Function-learning extension:** add one-dimensional and multivariate target functions with analytically known Fourier components. Vary the number, amplitude, and spacing of frequencies to measure which spectra the QNN can learn.
- **No leakage:** keep test inputs fixed and unused for training or model selection. Use validation data for stopping and hyperparameter choices.

## What to track

Each run should record configuration, random seeds, optimizer and learning-rate settings, parameter initialization, per-epoch training/validation losses, QFT process fidelity, state/output distribution distance, frequency accuracy, and elapsed time. Save the final parameters and enough metadata to reproduce the run. Record the simulator, shot count, and noise settings whenever applicable.

Use a small, explicit loss initially. For ideal unitary recovery, the primary loss is `1 - process_fidelity`. For supervised frequency training, use cross-entropy on frequency bins or a suitable distribution loss. Keep the metrics separate in logs and plots; do not blend circuit recovery and frequency accuracy into one score. If combining objectives is later useful, document and ablate the weighting.

## Visuals

The first experiment should produce a compact run report with:

1. Process-fidelity objective and independent training/validation state fidelity versus optimizer step.
2. Process fidelity versus step, with a reference line at 1.
3. Ideal and learned output probability distributions for held-out known-frequency inputs.
4. A confusion matrix for frequency-bin predictions.
5. A parameter trajectory against canonical QFT angles, with periodic-angle differences handled correctly.
6. A comparison across random seeds showing convergence traces and held-out scores.

Plots should label shot-based estimates and uncertainty when sampling noise is enabled. Save plots and machine-readable metrics under a run-specific output directory so experiments do not overwrite one another.

## Repository plan

### Phase 1 — Reproducible QFT recovery baseline

- Inspect and preserve useful work from `QNN/QNN.ipynb` and `QNN/DJ_alg.ipynb`; move reusable circuit and data logic into importable modules as it stabilizes.
- Define the QFT reference, parameterized ansatz, and explicit qubit/gate conventions.
- Implement the process-fidelity loss and independent held-out checks.
- Add seeded parameter initialization and a small deterministic fabricated dataset.
- Run a basic classical optimizer and save configuration, metrics, learned parameters, and plots.
- Establish acceptance criteria: a run reaches a high process fidelity from randomized initialization, held-out outputs agree with the reference, and repeated seeds report their success rate rather than only showing a best run.

### Phase 2 — Frequency task and experiment reporting

- Add fabricated periodic/phase-ramp inputs with known frequency bins.
- Measure output-distribution agreement, frequency accuracy, cross-entropy, and confusion matrices.
- Build a repeatable run-report workflow with loss and convergence visuals.
- Add validation-based stopping and ensure test data remain held out.

### Phase 3 — Robustness and optimizer study

- Compare optimizers under the same circuit, initialization distribution, data, stopping rule, and evaluation budget.
- Repeat each configuration across multiple seeds and report mean, spread, convergence failures, and compute cost.
- Add finite-shot and configurable-noise runs after the ideal baseline is stable.
- Keep the ideal simulator results as a control for separating optimizer issues from sampling/noise effects.

### Phase 4 — Function-learning research

- Add target functions with controlled Fourier spectra, including single-frequency and multi-frequency cases.
- Compare performance as function complexity, input dimension, ansatz depth, and training-set size change.
- Evaluate both function error on held-out points and spectral recovery error.
- Use the QFT recovery benchmark as a regression check when changing the circuit or training code.

## Initial project structure

```text
.
├── README.md
├── ARCHITECTURE.md
├── RESEARCH_PROTOCOL.md
├── pyproject.toml
├── configs/qft_recovery.toml
├── QNN/                    # existing exploratory notebooks
├── src/qftlab/             # reusable circuit/data/training/reporting modules
├── scripts/                # run, compare, and export utilities
├── slurm/                  # single-run and seed-array templates
└── results/                # generated run bundles; ignored by git
```

Keep notebooks for exploration and explanation; reusable experiment logic now lives in importable modules. See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries and tracking/cluster conventions, and [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for scientific controls, data schemas, and reanalysis needs.

## Quick start

From the repository root, install the package and run the local baseline:

```bash
python -m pip install -e .
python scripts/run_experiment.py --config configs/qft_recovery.toml
```

Each invocation writes a unique directory under `results/runs/` with the resolved config, manifest, seeded dataset, step metrics, initial/final/target parameters, final held-out predictions, plots, and an HTML report. `scripts/compare_runs.py` builds a portable summary table from those bundles:

```bash
python scripts/compare_runs.py
```

The baseline objective is exact process fidelity, so the small two-qubit run is practical on a workstation. The generated state data are diagnostic splits rather than the loss used in this first unitary-calibration protocol.

## MLflow tracking

MLflow is optional. Without a tracking URI, the runner creates complete portable local bundles. To log to a local tracking server, install the extra and start MLflow in one terminal:

```bash
python -m pip install -e '.[tracking]'
mkdir -p results
mlflow server --backend-store-uri sqlite:///results/mlflow.db --default-artifact-root results/mlflow-artifacts --host 127.0.0.1 --port 5000
```

Then in another terminal, point the runner at it:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python scripts/run_experiment.py --config configs/qft_recovery.toml
```

The run artifacts are logged to MLflow and also kept in the local bundle. Use `MLFLOW_TRACKING_URI` with an institution-approved server on the supercomputer; confirm compute-node and artifact-store access before long jobs. If the cluster cannot reach tracking, keep `tracking.mode = "auto"` and use the portable output on approved shared storage for later import.

## SLURM

The submission templates use the same runner. Adjust time/resources for local cluster policy and submit from the repository root:

```bash
sbatch slurm/run_one.sbatch
sbatch slurm/run_seed_array.sbatch
```

Set `QFTLAB_PYTHON` to the environment Python and `QFTLAB_OUTPUT_ROOT` to an approved writable/shared path when needed. The array template launches independent initialization seeds against the same fixed data split. Each task has its own run directory and records SLURM job metadata.

## Suggested first implementation choices

Use the simulator and optimizer already available in the environment where practical, and record exact dependency versions before comparing results. Keep the first circuit and objective small enough to inspect directly. Prefer an ideal, differentiable statevector/unitary path for the recovery baseline; introduce finite shots and hardware-oriented constraints only after the baseline is validated.

## Current status

The repository contains the original exploratory QNN and Deutsch–Jozsa notebooks, a config-driven QFT recovery runner, and a first supervised scalar-regression runner. The scalar MSE experiment uses a period-one `RY(2πx)` encoding on qubit 0, a QFT-layout trainable block, and Pauli-Z readout on qubit 1. Ten ideal-simulator initialization trials are documented in [METHODOLOGY.tex](METHODOLOGY.tex) and saved as portable run bundles. Supervised QFT imitation, finite-shot/noisy function runs, optimizer comparisons, and broader function-learning protocols remain planned work.
