# Research Protocol and Data Contract

This document specifies what the lab should measure and retain so its results remain scientifically interpretable and can be reanalyzed into new visualizations later.

## First separate the claims

The project has three related but different experiments. Keep their results and conclusions separate.

1. **Circuit calibration / unitary compilation.** Given the known two-qubit QFT and an ansatz with its gate structure, can randomized angles recover the target unitary? Process fidelity is an appropriate whole-operation score. This is a valuable implementation and optimizer sanity check, but the target unitary is supplied directly.
2. **Supervised QFT imitation.** Given input/output examples from the reference QFT, can the circuit match unseen states and distributions? Vary the number and kind of training states. Report held-out agreement and compare with whole-unitary fidelity to expose underconstrained solutions.
3. **Function learning.** Given classical inputs `x` and target values `f(x)`, can a quantum model with an explicit encoding and observable learn functions with controlled spectra? This is where frequency support, coefficient learning, data efficiency, and generalization become central.

Recovering the QFT with a circuit already laid out as the QFT does not by itself establish arbitrary-function learning. Treat it as a known-answer calibration before studying more general ansätze and target functions. In variational quantum models, the encoding often determines the accessible Fourier frequencies while trainable circuit parameters affect their coefficients; therefore record and vary the encoding as a scientific factor, not just the optimizer. See [Schuld, Sweke, and Meyer (2021)](https://doi.org/10.1103/PhysRevA.103.032430).

## Questions to settle before a large run

- What is the model's input encoding, data domain, normalization, output observable, and prediction scale?
- Which ansatz gates are fixed, which are trainable, and which degrees of freedom can express the target family?
- Is the training target a full unitary, output state, measurement distribution, expectation value, or scalar function value?
- What is the primary endpoint, what is a successful run, and at which checkpoint is test performance evaluated?
- Which factors are held fixed in optimizer comparisons: initialization distribution, data, seed list, steps, circuit evaluations, shots, noise, and stopping rule?
- Which baselines receive the same information and evaluation budget?

Put answers in a versioned experiment config/protocol. For the calibration experiment, define success by phase-insensitive process fidelity and held-out state/output agreement; compare learned angles only as a diagnostic because equivalent circuits may differ by periodic angles, global phase, or redundant parameterizations. For function learning, define a fixed target family and test split before optimizer tuning.

## Chosen classical feature encoding for the first function benchmark

There is no encoding that is best for every task. For our first **one-dimensional, periodic scalar-regression** protocol, use unitary angle encoding:

```text
x ∈ [0, 1)  →  RY(2πx)|0⟩ = cos(πx)|0⟩ + sin(πx)|1⟩
```

This choice matches the period-one Fourier target family, avoids amplitude-vector normalization, and is implemented by a unitary gate with inverse `RY(-2πx)`. Treat the domain as the circle `[0,1)`: the periodic endpoints are one input, so `x=0` and `x=1` must not be represented as distinct examples. In the initial two-qubit scalar model, encode `x` on qubit 0, initialize qubit 1 to `|0⟩`, apply the QFT-layout variational block, and predict with the Pauli-Z expectation on qubit 1. This readout is selected because the final SWAP makes qubit 0's Z population independent of the encoded input for this gate ordering. Scale targets to that observable's `[-1,1]` range before using MSE.

This is a reversible **encoding operation**, not a promise that a measured quantum state can reveal an arbitrary continuous input exactly. Encoded states are generally nonorthogonal, and measurement is probabilistic and changes the state. Keep the original classical `x` in each dataset and run artifact. Do not call the full learning/readout pipeline lossless.

Angle encoding is periodic, so define the input domain and scaling in each config. If a later task uses a nonperiodic closed interval where its endpoints differ, give it a separate encoding protocol (for example, restrict `RY(πx)` to `x∈[0,1]`) and test injectivity explicitly. Do not silently clamp, wrap, normalize, or quantize inputs. The number and placement of data reuploads control the accessible Fourier spectrum; begin with a single-frequency target the shallow model can represent, then increase spectral complexity and reupload depth as declared experimental factors. The relation between encoding generators and accessible frequencies is a central property of variational quantum models ([Schuld, Sweke, and Meyer, 2021](https://doi.org/10.1103/PhysRevA.103.032430)).

This encoding decision applies to the scalar function-learning protocol. It does **not** describe the QFT-recovery benchmark's random statevectors or its four phase-ramp states, which are quantum-state inputs rather than classical feature vectors.

The formal LaTeX methodology for the first supervised MSE function-learning experiment is maintained in [METHODOLOGY.tex](METHODOLOGY.tex). It specifies the model, seeded Fourier target data, train/validation/test rules, objective, evaluation, baselines, visual artifacts, and current implementation status.

## Synthetic benchmark suite

Use synthetic data first because its generating process, ground truth, and exact spectrum are known. Create a versioned generator for each family and record its full config and seed.

### QFT / frequency inputs

- Computational basis states and selected superpositions for circuit imitation.
- Phase-ramp and periodic amplitude states with known discrete frequency bins for the QFT task.
- Frequency cases with peaks at every bin, mixtures of bins, and boundary/degenerate cases.
- Controls including uniform distributions and deliberately uninformative predictions.
- Exact statevector probabilities as the noiseless target; sampled integer counts for finite-shot conditions.

Two qubits provide only four computational-basis output bins. Use this stage to validate conventions and metrics. To study richer frequency structure, increase qubit count or introduce a classical input function model with a clearly specified encoding; do not imply that a four-bin task demonstrates broad spectral learning.

### Scalar and multivariate function targets

Generate targets as finite Fourier series with explicitly known frequencies and complex/real coefficients. Include constant, single-frequency, sparse multi-frequency, dense low-band, and mixed low/high-frequency cases. Vary bandwidth, coefficient dynamic range, phase, input dimension, and sample count independently. Include train/validation/test points from both the same domain distribution and a declared extrapolation region when extrapolation is a question.

Save both analytic target coefficients and pointwise values. Include noiseless targets first, then optional observation noise with its distribution and seed recorded. Use uniform sampling for the base study and add targeted/adaptive sampling only as a separately labeled protocol.

### Baselines and controls

- Exact reference circuit / ideal target outputs as a correctness ceiling.
- Identity, random initialization, and a declared random-circuit control.
- Multiple random seeds; report all runs and convergence failures.
- A classical Fourier regression baseline using the same target basis or frequency access.
- A small classical neural baseline with comparable input data and tuning budget for function-learning claims.
- For noisy tasks, compare ideal prediction, noisy quantum samples, and the corresponding statistical baseline.

Classical baselines are not needed to validate circuit construction, but are essential context before making claims about quantum learning advantage. Ensure they receive the same train/test examples and disclose differences in feature access and compute budget.

## Data to retain

Keep the following information at the stated level. Store large arrays as artifacts and index summaries/metrics in MLflow.

### Per experiment / protocol

- Protocol name and version; target family equation and parameters; domain and units; data encoding and input normalization; output observable/measurement rule; ansatz and gate conventions.
- Dataset generator name/version, all generator seeds and settings, exact split membership or a stable split ID, and target spectrum/coefficients.
- Metric definitions, aggregation rules, stopping rule, and compatibility fields for fair comparisons.

### Per run

- Resolved config, code revision, environment/dependency versions, simulator/backend, precision, noise and shot settings.
- Initialization seed, data seed, sampler seed, optimizer settings, learning-rate schedule, stopping reason, wall time, and resource/evaluation counts.
- Initial, best-validation, and final parameter vectors; optimizer checkpoint/state when continuation is needed.
- Scheduler IDs/resources and exit status for cluster runs.

### Per training step or checkpoint

- Step, elapsed time, train and validation losses, and all named diagnostic metrics.
- Parameter vector at each step for small ansätze; at minimum every step for the 2-qubit circuit. For larger models, use a declared checkpoint cadence plus final/best values.
- Learning rate, gradient norm, and optimizer diagnostics when available. Keep these optional fields clearly marked rather than silently omitting them.
- Number of circuit evaluations and shots consumed, not just epochs. This helps compare optimizers with different evaluation costs.

### Per fixed evaluation example

At initial, selected training checkpoints, best-validation, and final parameters, save the fixed evaluation input and enough outputs to reconstruct metrics:

- stable sample ID, split, input values/state ID, target label/value, and exact target output;
- ideal model probabilities or expectation values;
- sampled integer counts and shots when sampling is used;
- predicted class/frequency and any derived score.

Retaining per-example outputs makes later residual, calibration, confusion, and subgroup plots possible without rerunning a potentially nondeterministic experiment. Store all checkpoints only when storage is manageable; otherwise predeclare a compact checkpoint schedule and always include the final and selected checkpoints.

## Storage contract

Use MLflow for searchable run parameters, scalar metric histories, tags, and artifacts. Keep structured files as the portable source for reanalysis; never make a plot image the only copy of its data.

- **Small run histories:** JSONL for per-step metrics and parameter values, with stable field names and explicit units. Parquet/CSV can be added for larger per-example tables when the optional table dependencies are installed.
- **Numerical tensors:** `.npz` (without object/pickle arrays) for small fixed-shape amplitudes, probabilities, parameter trajectories, or count tensors; a JSON sidecar records axis names, shape, dtype, basis ordering, and units.
- **Large/chunked arrays:** choose a chunked format such as Zarr or HDF5 only when scale justifies it, and record chunk dimensions and format version.
- **Human-readable summaries:** JSON manifest/summary and resolved TOML/JSON config.
- **Visual artifacts:** SVG/PNG plus a report HTML; plots are generated from the structured records.

Recommended table keys are `run_id`, `protocol_version`, `seed`, `step`, `split`, and `sample_id` as applicable. Use one row per metric observation or a documented wide schema; avoid ambiguous column names. Store both integer counts and normalized probabilities when shots are involved. Store complex amplitudes as real/imaginary components or a documented complex array format, and define computational-basis bit ordering explicitly.

Do not commit generated run data to Git by default. Keep small canonical fixtures in the repo for correctness checks, and place run artifacts in MLflow-approved artifact storage or portable run bundles. A manifest should link MLflow IDs to exported file paths and include checksums for important data artifacts.

## Make visualization open-ended

The key is not to guess every plot in advance; it is to preserve normalized, labeled observations and predictions. Provide a small analysis API that can load runs into tables/arrays and filter by run, seed, step, split, target frequency, optimizer, or protocol. Custom notebooks/scripts should be able to access the same stable data files directly.

Initial views should include:

- loss, validation loss, process fidelity, and frequency accuracy versus optimizer step;
- convergence traces overlaid by seed and final-score distributions;
- predicted versus target distributions for each frequency case, including sampled-count uncertainty;
- frequency confusion matrices and per-frequency accuracy;
- target and learned function curves over a dense grid, with training points and residual plots;
- true versus learned Fourier coefficients and spectrum error;
- parameter and gradient trajectories, with circular differences for periodic angles;
- resource curves versus circuit evaluations, shots, and wall time.

Every view should be reproducible from run IDs and raw artifacts, with axes/units and protocol labels. Keep a report's plot selection customizable, but make the underlying run data independent of that selection. Provide both a standard report generator for common summaries and a notebook-friendly loader for ad hoc visualizations.

## Suggested acceptance criteria before optimizer comparisons

- Reference QFT and parameterized circuit conventions pass known-answer checks.
- The exact QFT parameter setting achieves the declared target score under the correct qubit ordering and global-phase convention.
- A seeded generated dataset reproduces byte-for-byte or has a verified stable content hash.
- Run bundles distinguish target, train, validation, and test data and contain raw per-example outputs for fixed evaluation cases.
- A failed/interrupted run is visibly marked and retains its config and partial history.
- A report can be rebuilt from logged data alone and produces the same values as the machine-readable summary.
- Multi-seed comparison includes every seed, convergence failures, uncertainty, and cost budget.
- QFT calibration, supervised QFT imitation, and scalar function learning are labeled as distinct protocols.
