# Repository Architecture

This document proposes a structure for a small, reproducible quantum machine learning research lab. MLflow is the searchable tracking layer and visual run browser; a portable run bundle is the durable export and recovery path. The same experiment code runs on a workstation or inside a SLURM job.

For benchmark design, data fields, storage formats, scientific controls, and analysis flexibility, see [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

The architecture should make it straightforward to answer three questions for every figure or metric:

1. What exact question and configuration produced it?
2. Can the run be reproduced or continued?
3. Is the comparison scientifically fair to the other runs shown beside it?

## Principles

- **Separate scientific layers.** Circuit definitions, synthetic data, objectives, optimization, and reporting have distinct interfaces.
- **Keep a complete record.** MLflow indexes parameters, metrics, tags, and artifacts; each run also exports its resolved configuration and a manifest with enough provenance to interpret it outside MLflow.
- **Make runs immutable.** Never silently overwrite a completed run. A rerun receives a new run ID and records its seed and configuration.
- **Keep exploratory work useful.** Notebooks remain for investigation and narrative, while reusable code lives in importable modules.
- **Compare like with like.** Results record evaluation budgets, seeds, simulator and shot settings, and metric definitions. Comparisons state which fields must match.
- **Prefer inspectable formats.** Use YAML or TOML for human-authored configuration, JSON for manifests and summaries, JSONL or CSV for metrics, and PNG/SVG plus HTML for visuals.
- **Support local and cluster execution.** The training entry point stays independent of the scheduler. Local runs need no SLURM; cluster runs use thin submission scripts and the same resolved config format.
- **Keep portable outputs.** Reports and run exports should remain readable if the tracking server is unavailable or the project moves to another system.

## Proposed repository layout

```text
.
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml              # package metadata and pinned/managed dependencies
├── configs/
│   ├── qft_recovery.toml       # canonical baseline configuration
│   └── function_learning.toml  # later benchmark configuration
├── data/
│   ├── README.md               # dataset conventions and provenance
│   └── fixtures/               # tiny committed examples, if useful
├── notebooks/
│   ├── exploration/            # exploratory work; promote stable logic to src
│   └── reports/                # optional notebook narratives linked to run IDs
├── src/qftlab/
│   ├── circuits/               # canonical QFT and parameterized ansatz
│   ├── data/                   # seeded synthetic input and target generators
│   ├── objectives/             # unitary, state, distribution, and function losses
│   ├── training/               # optimizer-independent training loop and callbacks
│   ├── evaluation/             # fidelity, frequency, and generalization metrics
│   ├── experiments/            # compose components into named benchmark protocols
│   ├── reporting/              # plots, run summaries, and report generation
│   └── utils/                  # seeds, versions, serialization, and shared helpers
├── scripts/
│   ├── run_experiment.py       # config + overrides -> one immutable run bundle
│   ├── export_mlflow_run.py    # MLflow run -> portable result bundle
│   └── compare_runs.py         # summarize portable runs and plots
├── tests/
│   ├── test_circuit_conventions.py
│   ├── test_objectives.py
│   ├── test_data_reproducibility.py
│   └── test_run_bundle.py
├── results/                    # generated exports/reports; git-ignored
│   └── runs/
├── slurm/
│   ├── run_one.sbatch           # single configured experiment
│   └── run_seed_array.sbatch    # independent seeds/config variants
└── .gitignore
```

The existing `QNN/` notebooks can first be retained in place. Moving or renaming them is a cleanup step, not a prerequisite for the architecture. Likewise, directories should be added as work reaches them rather than created empty just to match this diagram.

## Module responsibilities and boundaries

### Circuits

`circuits/` defines the reference operation and parameterized circuit. The QFT convention belongs next to the implementation: logical qubit order, physical wire order, controlled-phase angle and sign, swap behavior, and global phase. Provide a way to obtain the ideal operator or ideal output state for evaluation. Do not let plotting or dataset code construct private variants of the QFT.

The first ansatz should make its trainable degrees of freedom explicit. Keep a canonical parameter vector and a mapping from parameters to gates so periodic angles can be compared correctly. Tests should establish that the reference parameter setting reproduces the reference QFT up to global phase.

### Data

`data/` owns deterministic generators, not precomputed random state hidden in notebooks. Each generator accepts an explicit seed and a configuration, and returns examples plus labels/targets and provenance. Train, validation, and test partitions are stable for a given dataset version and seed. Test data must not influence training, early stopping, or optimizer selection.

Generated samples may be saved inside a run bundle when needed for exact replay. Small canonical fixtures can be committed; larger generated datasets should record generator name, version, parameters, and seed instead of expanding the repository with redundant files.

### Objectives and evaluation

`objectives/` contains quantities used to update model parameters. `evaluation/` contains all reported scores, including the training objective where applicable. This separation avoids mistaking a low training loss for successful algorithm recovery.

For the first milestone, report at least:

- phase-insensitive QFT process fidelity;
- held-out output-state or probability-distribution agreement;
- known-frequency top-1 accuracy and cross-entropy;
- function error and spectral error when general function benchmarks are added.

Every metric has a documented formula, input assumptions, range, and aggregation rule. Include ideal unitary recovery and supervised frequency performance as separate outcomes. If a composite objective is explored later, record its weights and keep the uncombined metrics in every report.

### Training and experiment protocols

`training/` owns a common loop, optimizer adapter, callbacks, and checkpoint cadence. `experiments/` defines a benchmark protocol: which circuit, data generator, objective, optimizer settings, stopping rule, and evaluations are composed together. Experiment-specific choices should be configuration, not scattered constants.

The first runner should support one config file plus explicit command-line overrides. At launch it resolves all defaults, writes the resolved configuration into the run record, and then trains. The resolved configuration, rather than the original shorthand config alone, is the reproducibility record. The runner logs through a small tracking interface so training code is not coupled to MLflow calls throughout the scientific modules.

## MLflow and durable run exports

Use one MLflow experiment per benchmark family or scientific question, then use tags for protocol version, ansatz, execution environment, and comparison group. Log scalar hyperparameters as MLflow parameters; log per-step loss, fidelity, frequency, and timing metrics with a consistent step; log resolved config, parameter vectors, plots, and HTML reports as artifacts. Keep structured parameter vectors as JSON/NPZ artifacts, with readable scalar tags for a few named angles when useful. Use the MLflow UI for searching, filtering, metric plots, and quick comparisons. Keep custom reports for research-specific visualizations such as frequency confusion matrices, held-out distributions, and multi-seed uncertainty views. MLflow supports run metadata, parameter/metric logging, artifact storage, and a run-exploration UI. ([Tracking](https://mlflow.org/docs/latest/tracking/), [Tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/))

Use two execution profiles:

- **Local development:** run the normal CLI with a local MLflow tracking URI/server. Keep the database and artifacts under a user-selected ignored directory, not in the source tree. This is convenient for one-user experiments and prototyping.
- **Shared/cluster tracking:** point local and SLURM clients at an institution-approved tracking server. Use a database-backed metadata store and an artifact store reachable by the server/clients as configured. MLflow documents SQLite as its default and recommends a relational database such as PostgreSQL or MySQL for higher-concurrency deployments; artifact storage is configured separately from run metadata. ([Backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [Artifact stores](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/))

Do not assume compute nodes can reach a tracking service or object store. Before long jobs, confirm cluster policy and network/filesystem access. If compute nodes cannot reach MLflow, make that an explicit offline mode: write the complete portable run bundle to approved scratch/shared storage and import/log it later through a documented ingestion step. Do not silently drop metrics or report a run as tracked when it was not. Avoid putting credentials in configs, job scripts, or logged environment dumps.

### Running locally and on SLURM

Keep `scripts/run_experiment.py` scheduler-neutral. It accepts the resolved experiment config, an optional seed/replicate override, and output/tracking settings. A thin SLURM script loads the agreed software environment, stages code/config as required by local cluster practice, records scheduler identifiers as tags, and invokes the same runner. An array script maps each array task to a deterministic seed or config row and gives every task a separate MLflow run and unique output path. SLURM job arrays are designed for collections of jobs with identical resource requirements and provide per-task identifiers, which suits independent seed sweeps. ([SLURM job arrays](https://slurm.schedmd.com/job_array.html), [sbatch](https://slurm.schedmd.com/sbatch.html))

Record scheduler job ID, array task ID, cluster/partition, requested CPU/GPU/memory/time, and exit status as tags/metadata. Capture stdout/stderr under a run-specific path as well as the MLflow record. Set resource requests in the submission layer, not in the scientific configuration, except where a hardware choice changes the method and must be recorded as part of the experiment.

Local and SLURM executions must share the same code revision, config semantics, seed handling, metrics, and output naming. The only intended differences are declared execution resources and environment metadata. Use SLURM for sweeps, large shot budgets, noisy simulation, deeper circuits, and long multi-seed studies; preserve a local profile for quick development and baseline runs.

### Portable run bundle: durable unit of research output

Each invocation creates a fresh portable bundle and, when MLflow is configured, logs the same files as artifacts:

```text
results/runs/20261016T120000Z_qft-recovery_seed-0042_a18f2c/
├── manifest.json              # identity, status, config hash, software and data provenance
├── config.toml                 # input configuration
├── config.resolved.json       # effective settings and run overrides
├── dataset.npz                 # seeded train/validation/test and frequency data
├── dataset_manifest.json       # array shapes, dtypes, seed, and basis-order convention
├── metrics.jsonl               # per-iteration metrics and parameter trajectories
├── summary.json                # final validation/test metrics and optimizer status
├── evaluation_outputs.npz     # held-out inputs, targets, predictions, and probabilities
├── parameters/
│   ├── initial.json
│   ├── final.json
│   └── target_qft.json
├── artifacts/
│   ├── loss.png
│   ├── process_fidelity.png
│   ├── frequency_confusion.png
│   ├── frequency_distributions.png
│   ├── parameter_trajectory.png
│   └── report.html             # run page linking figures and metrics
```

The exact timestamp, labels, and short config hash can vary, but the run ID must be unique and stable within the bundle. Keep both a human-readable label and an unambiguous generated ID. A run that fails should still preserve its manifest, resolved configuration, logs, and failure status.

`manifest.json` should capture:

- run ID, benchmark name/version, creation time, completion status, source revision, and source fingerprint;
- configuration hash and location of the resolved configuration;
- all random seeds, including data, parameter initialization, and simulator seeds;
- dependency versions, Python version, operating system, and relevant accelerator/backend;
- circuit and dataset versions, split identifiers, and generator settings;
- optimizer name/settings, stopping rule, number of steps, and evaluation budget;
- simulator, precision, shot count, and noise model where applicable;
- MLflow experiment/run IDs and artifact paths, if tracked;
- artifact paths and final metric summary.

Metric records should have stable field names and include step/epoch, split, objective, metrics, and elapsed time. Store full precision in machine-readable output; round only in plots and tables. Keep training and validation series, and evaluate the held-out test set at the declared final checkpoint (or a separately declared best-validation checkpoint).

## Visual results experience

Use MLflow's UI as the primary live index of tracked runs. Each run includes a human-readable HTML report artifact with configuration summary, final scores, convergence plots, and parameter values. `scripts/compare_runs.py` generates local HTML/CSV and trajectory/held-out-score plots from portable bundles. The UI and reports expose benchmark, ansatz, optimizer, seed, status, final process fidelity, frequency accuracy, and run date.

For scientifically meaningful side-by-side views, add comparison pages that group compatible runs and show:

- loss and process-fidelity trajectories with the same axes and evaluation budget;
- final-score distributions across seeds (with each seed visible, not only a mean);
- frequency confusion matrices and representative held-out distributions;
- elapsed time or objective evaluations when comparing optimizer efficiency.

Reports are views over immutable run records. Rebuilding them must not alter experiment outputs. A `latest.json` shortcut may point to the newest successful run for convenience, but must never replace the run archive.

Generated result directories and local MLflow stores should be git-ignored by default. Commit source, configs, documentation, and carefully selected small reference artifacts. For results that need to be shared, retain the MLflow record and archive complete portable bundles in approved storage; preserve the manifest and source revision either way.

## Scientific comparison rules

Before comparing optimizer or ansatz variants, define a comparison group and validate that runs agree on the factors intended to be controlled: target operation, data/splits, initialization distribution, seed set, shot/noise budget, stopping rule, and evaluation budget. Store deliberate differences as labeled factors. A comparison script should warn or refuse to pool runs with incompatible protocol fields.

Use repeated seeds and show both individual outcomes and aggregate uncertainty. Report convergence failures and incomplete runs. Avoid choosing the most favorable seed as the headline result. Separate training, validation, and final test metrics in both raw files and visual reports. Preserve enough run data to reproduce figures from the underlying metrics rather than manually editing plot values.

## Build order

1. **Repository skeleton:** package metadata, `src/qftlab` layout, config convention, MLflow dependency/profile, and result/store ignore rules. *(Implemented.)*
2. **Known-answer circuit checks:** implement reference/ansatz and verify gate ordering, global-phase handling, and process fidelity. *(Circuit and objective implemented; checks still needed.)*
3. **Local end-to-end run:** deterministic configuration and generator, training, evaluations, MLflow parameters/metrics/artifacts, and portable export. *(Runner and bundle implemented; one local seed completed.)*
4. **SLURM smoke path:** submit one short run through the same entry point, confirm tracking/artifact connectivity or explicit offline export, and capture scheduler metadata. *(Templates implemented; cluster-specific validation remains.)*
5. **Visual report:** per-run report and comparison report generated from logged artifacts/metrics; verify they identify the underlying run IDs. *(Report generators implemented and used for the first local run.)*
6. **Multi-seed protocol:** launch repeated seeds locally or as a job array, compute transparent aggregate summaries, and validate compatibility before plotting comparisons. *(Seed-array template and summary plots implemented; compatibility checks remain.)*
7. **Frequency benchmark:** add controlled frequency states and their separate task metrics and visuals.
8. **Scalar function-regression protocol:** use the documented period-one angle encoding, Pauli-Z readout, and MSE on sampled `(x, y)` pairs; log exact target spectra only for evaluation. *(Initial MSE runner and ten-seed ideal run implemented; broader target families remain.)*
9. **Optimizer/noise studies:** vary one factor at a time or declare a designed sweep; retain the ideal baseline as a control.
10. **General function suite:** add analytically controlled spectra and track both input-output error and frequency recovery.

## First milestone acceptance criteria

The baseline is ready for optimizer research when a new user can run a documented local command or SLURM job from a clean checkout and receive:

- a run directory with resolved settings, provenance, metrics, final parameters, and status;
- plots showing objective and process fidelity through training;
- held-out QFT output comparisons and a frequency score where the frequency dataset is enabled;
- a report page discoverable from MLflow or a portable local export;
- a repeatable multi-seed summary that exposes failed or incomplete runs;
- correctness checks demonstrating that the canonical parameter setting matches the reference QFT and that run data can be parsed without the training process.

This architecture uses MLflow for interactive tracking while retaining a portable file bundle as the durable scientific record. The repository has ideal unitary recovery and an initial ideal periodic scalar MSE runner; broader supervised imitation and function-learning protocols remain future phases. The tracking server and artifact-store deployment should be chosen with the supercomputer's access, security, and storage rules in mind; the experiment code should not depend on one specific cluster layout.
