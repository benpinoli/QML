# Hypothesis tree

This human-readable view is backed by [`hypotheses/HYPOTHESIS_TREE.json`](hypotheses/HYPOTHESIS_TREE.json), the machine-readable source of record. The JSON schema is [`hypotheses/HYPOTHESIS_TREE.schema.json`](hypotheses/HYPOTHESIS_TREE.schema.json). Keep node IDs stable as evidence accumulates; add experiment records instead of erasing earlier observations.

## Root question

**Does a randomly initialized QFT-layout circuit recover the goal QFT, or can it perform the task without recovering that full unitary?** These are separate outcomes. Direct QFT recovery is assessed on the full operation; task performance is assessed through held-out prediction error and frequency recovery. The current MSE runs use ideal statevector expectations and do not include process fidelity, PINN residuals, or frequency labels in the loss. Neither outcome alone establishes arbitrary-function learning.

### Central experimental question

The lab is studying two possible successes:

1. **Algorithm recovery:** randomized parameters converge to the canonical QFT, measured with phase-insensitive full-unitary process fidelity.
2. **Task success without recovery:** the circuit predicts the target function and recovers its frequencies, even while process fidelity remains low.

**Existing evidence for both outcomes:** the direct QFT-recovery pilot (seed 42) rose from process fidelity `0.0921` to `0.999999999999979` in 17 optimizer iterations (176 objective evaluations). This shows convergence for one randomized initialization, not a convergence guarantee. Separately, the zero-offset single-frequency MSE runs reached near-zero task error with low QFT fidelity, showing task success without full-unitary recovery for that task. The per-run QFT report is [here](results/runs/20261017T000333116098Z_qft-recovery_seed-0042_b7a380cf/artifacts/report.html).

The goal is to understand these outcomes, not to identify the best-performing ansatz. Preserve the QFT-layout baseline and treat encoding and circuit depth as explicit experimental factors. A fixed, non-learned encoding sweep can keep the seven-parameter trainable block unchanged, while still changing the overall function class. Repeated data encoding between blocks changes the input-dependent circuit; if the number of trainable blocks also changes, use a matched-depth control to separate the effects. The next design choice is **pending lab discussion**.

### A. Is the encoding reversible?

**Question:** Does the feature map preserve encoded information under its inverse, and what can measurements recover?

- **Supported:** On the period-one domain, `x -> RY(2*pi*x)|0>` is injective and the encoding gate has inverse `RY(-2*pi*x)`. The domain is `[0,1)`; `x=1` is excluded because it represents the same periodic point as `x=0`.
- **Limit:** Reversibility describes the encoding operation. One measurement does not generally decode a continuous input exactly because distinct encoded states are nonorthogonal and outcomes are probabilistic.
- **Next experiment:** Encode a fixed grid, apply the inverse, and record recovered-state fidelity.

### B. Why was the first function fit not perfect?

**Experiment:** Target `g(x)=0.10+0.55*cos(2*pi*x)+0.25*sin(2*pi*x)` over `x in [0,1)`. Encode with `RY(2*pi*x)` on qubit 0, apply the trainable `U`—controlled-phase—`U`—SWAP block, and read out ideal Pauli-Z on qubit 1. Train on 64 examples with empirical MSE only. Ten initialization seeds (42–51) share the fixed data seed 314159.

**Observed:** Mean test MSE is **0.01039**, `R2=0.94351`, and dominant-frequency accuracy is **10/10** at `k=1`. The constant baseline has test MSE **0.18989**. Classical Fourier regression supplied with `k=1` has test MSE approximately zero; it is an informed reference, not an equal-information frequency-discovery baseline.

This is a useful fit, but not a perfect one. A **provisional next milestone** is test `R2 >= 0.99` (approximately sample-normalized MSE `<= 0.01`) and correct `k=1` support. That criterion was not preregistered for the completed pilot. Because the target is noiseless, first determine whether this circuit can represent it exactly.

For seed 42, dense-grid Fourier coefficients `(DC, cos k=1, sin k=1)` are `(0.10, 0.55, 0.25)` for the target and approximately `(0.00, 0.5213, 0.2516)` for the QNN. The missing DC offset appears to be the largest systematic error, with a smaller cosine-amplitude error. Mean absolute error over the period is **0.10** and maximum absolute error is about **0.1288**. First vary the target offset, including zero, while holding other factors fixed.

**Structural explanation now supported:** After the final SWAP, measuring `Z` on qubit 1 is equivalent to measuring `Z` on qubit 0 immediately before the SWAP. The first `U` gate prepares qubit 1 in a superposition. The controlled-phase gate then applies either the identity or a phase rotation to qubit 0, conditioned on qubit 1. Since those qubit-1 branches are orthogonal, the effective observable seen by encoded qubit 0 is a weighted sum of two unitary conjugates of Pauli `Z`. Both terms are traceless, so their weighted sum is traceless. Averaging over the uniform period-one encoded circle therefore gives zero model mean: this model cannot produce the target's DC offset of 0.10. Under uniform evaluation, that alone gives an MSE floor of `0.10^2 = 0.01` if the harmonic part is otherwise exact.

This also explains why the QNN curve sits consistently below the target by about 0.1. The exact target has an offset of 0.1; the learned model's mean is zero. The remaining small harmonic mismatch is also understood: classical least squares restricted to `a*cos(2*pi*x)+b*sin(2*pi*x)` with **no intercept** gives coefficients `(0.5212828060, 0.2515603834)` and train/validation/test MSE matching the QNN to numerical precision. The current L-BFGS-B run found that constrained empirical optimum. Its residual is therefore primarily an output/readout limitation, not evidence that the optimizer failed.

**Next discriminating experiments:**

1. **Offset control — completed proof of concept:** Train on the zero-offset version of the same `k=1` target. Five initialization seeds (42–46) gave mean test MSE **4.04e-17** (population SD **4.09e-17**) and recovered `k=1` in **5/5** trials. The runs used the same fixed seed and input points as the offset group; only the target labels changed, by exactly 0.10. Dense-grid mean absolute error was at most `9.74e-9` across these trials. This supports the explanation that the original gap was the unavailable DC term. Because this is five initializations on one dataset, it is a paired proof of concept, not evidence of robustness across data samples.
2. **Add a bias/readout degree of freedom:** Compare a trainable scalar output bias against a modified circuit/readout that can produce a nonzero mean. Keep MSE as the sole objective and label the added model factor.
3. **Sample-size check:** Use nested training sizes and multiple data seeds to measure the finite-sample harmonic-coefficient mismatch above the structural DC floor.
4. **Optimizer comparison:** Revisit optimizer comparisons after the bias/readout limitation is addressed, so optimizer capability is not confounded with a missing output degree of freedom.

The zero-offset group and its per-run reports are at [the grouped report](results/periodic-function-mse-zero-offset-z1-readout/index.html); the machine-readable aggregate is [summary.json](results/periodic-function-mse-zero-offset-z1-readout/summary.json). The QFT process-fidelity diagnostic remains far from 1 and varies across seeds, even while function MSE is effectively zero. This is direct evidence that learning this function does not require recovering the full QFT unitary.

**Different-dataset check — preliminary success:** We repeated the zero-intercept experiment on five independent synthetic datasets, with five parameter initializations per dataset (25 runs total). Every dataset retains the same target, `g(x)=0.55 cos(2*pi*x)+0.25 sin(2*pi*x)`. This is **one non-DC frequency, `k=1`**: the sine and cosine coefficients are two quadrature components at the same frequency. All 25 runs recovered the correct dominant frequency (`k=1`), and test MSE ranged from **5.42e-19** to **1.60e-16** (mean **3.69e-17**). QFT process fidelity remained low: mean **0.1397**, maximum **0.5440**, with the canonical QFT corresponding to 1. These results preliminarily support function-fit stability across the tested data samples while the learned circuit remains unlike the full QFT. The five data seeds are still a small sample, and dominant-frequency accuracy does not test multi-frequency recovery.

The [multi-dataset report](results/periodic-function-mse-zero-offset-multidataset-z1-readout/index.html) puts the lowest-MSE and highest-MSE runs first, links every run report, and compares MSE against QFT process fidelity. Next retain zero intercept and add targets with two or more non-DC frequency components. Measure full spectral support recovery as well as function MSE and QFT process fidelity; do not infer multi-frequency capability from the current single-frequency result.

The first qubit-0 readout pilot was independent of `x` after the final SWAP and converged to the constant predictor. It remains saved as a readout-control pilot and is excluded from the valid qubit-1-readout group.

### C. Why do only some circuit parameters move?

For seed 42, shortest signed angular changes were:

| Parameter | Wrapped change |
| --- | ---: |
| `u0_theta` | +1.527 rad |
| `u0_lambda` | −0.732 rad |
| `u1_theta` | −0.564 rad |
| `controlled_phase` | −0.104 rad |
| `u0_phi` | approximately 0 |
| `u1_phi` | approximately 0 |
| `u1_lambda` | 0 |

Across ten seeds, mean absolute wrapped changes were largest in `u0_theta` (0.919 rad), `u0_lambda` (0.291 rad), and `u1_theta` (0.227 rad). No raw initial/final pair crossed zero in these ten trials. The optimizer does not constrain signs. Also, angle sign is not a physical invariant because rotations are periodic and Euler decompositions can be non-unique.

**Candidate explanations and experiments:**

- Some angles may have low prediction/loss sensitivity. Log per-parameter gradients and prediction Jacobians at initialization, intermediate checkpoints, and final values; perturb each angle individually.
- Some angles may be redundant for this scalar readout. Freeze or ablate one parameter at a time and inspect prediction-Jacobian singular values.
- L-BFGS-B may stop while other directions could improve the fit. Record gradient norms and stopping conditions; continue with tighter tolerance and compare optimizers.

`parameters.png` now shows initial and final values, canonical QFT parameters in green, and shortest signed change modulo `2*pi` in a second panel. The canonical vector is only a visual reference; the function-learning loss does not pull parameters toward it.

**Why does `u0_theta` usually move most?** The circuit order is feature encoding `RY(2*pi*x)` on q0, then trainable `U` on q1, controlled phase, trainable `U` on q0, final SWAP, and measurement of `Z1`. Thus `u0_theta` belongs to the second trainable `U`, on wire 0, just before the final SWAP; `u0` means wire 0, not “the first U.” In the 25-run zero-offset sweep, it had the largest wrapped angle change in **24/25** runs (mean absolute wrapped change **1.207 rad**). A post-hoc central finite-difference check at initialization found a median absolute training-loss derivative of **0.493** for `u0_theta`, versus **0.081** for `u0_lambda`, **0.046** for controlled phase, and **0.027** for `u1_theta`; derivatives for `u0_phi`, `u1_phi`, and `u1_lambda` were approximately zero. This is evidence that the loss is initially much more sensitive to `u0_theta` on these trials, but angle movement itself is an optimizer step, not a direct sensitivity measure.

The optimizer does **not** use backpropagation here: `scipy.optimize.minimize` is called without a gradient function, so L-BFGS-B estimates derivatives numerically and uses its quasi-Newton update and line search to choose steps. The small finite-difference probes are deterministic coordinate evaluations, not random perturbations. Randomness is used to initialize the seven angles; the target, loss evaluations, and optimizer steps are otherwise deterministic for a fixed run. Since this scalar target has only two non-DC coefficients but the circuit has seven angles, different parameter combinations can produce nearly the same function. To distinguish sensitivity from redundancy, next log gradients and prediction Jacobians through optimization, inspect Jacobian singular values, and freeze/ablate parameters one at a time.

The post-hoc per-parameter analysis is saved as [JSON](results/periodic-function-mse-zero-offset-multidataset-z1-readout/parameter_sensitivity.json) and [plot](results/periodic-function-mse-zero-offset-multidataset-z1-readout/parameter_sensitivity.png). Recompute it with `python scripts/analyze_parameter_sensitivity.py`.

### D. How does frequency complexity affect learning?

The initial zero-offset target had **one nonconstant frequency**, `k=1`, represented by cosine and sine coefficients, and no DC term. The original offset pilot had the same `k=1` plus DC `k=0`. Its 25 trials across five independent data seeds recovered the dominant `k=1`; that result did not test multi-frequency discovery or full frequency-support recovery.

**Completed nested-frequency experiment:** We kept zero intercept, used the same sampled inputs (data seed 314159), retained the existing coefficients, and added one higher harmonic at each step. Ten random initializations were run for each target:

| Distinct frequencies | Added target bins | Mean test MSE | Dominant `k=1` correct | Support precision | Support recall | Mean QFT process fidelity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 (`k=1..2`) | `k=2`: `(a_cos,a_sin)=(0.16,0.08)` | 0.01627 | 10/10 | 1.00 | 0.50 | 0.0883 |
| 3 (`k=1..3`) | plus `k=3`: `(0.10,0.05)` | 0.02246 | 10/10 | 1.00 | 0.33 | 0.0900 |
| 4 (`k=1..4`) | plus `k=4`: `(0.07,0.035)` | 0.02660 | 10/10 | 1.00 | 0.25 | 0.0890 |

The support metric uses a dense-grid Fourier amplitude threshold of `1e-3`. All 30 runs predict support `{k=1}` only: the dominant-frequency score remains perfect because `k=1` is the largest component, while support recall falls to `1/N`. This is the important distinction between detecting the largest frequency and learning all target frequencies. The increasing MSE is consistent with the additional components being missed. QFT process fidelity remains low and does not rise with function complexity.

**Why this happens:** the result points to the single data upload, not a two-qubit ceiling. The input is encoded once as `|ψ_x⟩ = cos(πx)|0⟩ + sin(πx)|1⟩` on `q0`; `q1` starts in `|0⟩`, and every gate after encoding is independent of `x`. Pull the measured observable backward through those gates and take its matrix element on `q1`. This leaves a fixed 2×2 effective observable on `q0`. Its expectation has the form `f_QNN(x) = c0 + c_cos cos(2πx) + c_sin sin(2πx)`. Thus this exact architecture can express DC and `k=1`, but no `k>=2`, regardless of how the seven trainable angles are optimized. Adding an unencoded ancilla or more `x`-independent gates alone would not lift that limit. Quantum-model Fourier analyses likewise identify data encoding and its repetitions as the source of accessible frequencies ([Schuld, Sweke & Meyer, 2021](https://arxiv.org/abs/2008.08605)).

This also explains both plateaus that have come up: the earlier single-frequency target with intercept `0.10` had a **0.01 DC floor** because this readout's output has zero period average; the newer zero-intercept multi-frequency targets have missing-harmonic floors. Under a uniform one-period MSE, omitting a harmonic contributes `(a_k²+b_k²)/2`, so the predicted floors are `0.01600` for `k=1..2`, `0.02225` for `k=1..3`, and `0.0253125` for `k=1..4`. The observed means (`0.01627`, `0.02246`, `0.02660`) are close. Small excess can come from imperfect fitting of the reachable fundamental and finite test sampling. A flat loss here is consistent with an expressivity ceiling; it does not by itself show that two qubits are insufficient or that the optimizer is stuck.

**Completed data re-upload experiment:** We kept the circuit at two qubits and compared one through four `RY(2πx)` uploads. The three zero-intercept targets, input splits (data seed `314159`), training objective, and ten initialization seeds were paired. Each upload adds one trainable QFT-layout block (7 parameters), so this experiment changes both input bandwidth and trainable depth. MSE and support recall are shown as `mean test MSE / mean frequency-support recall`:

| Target frequencies | 1 upload | 2 uploads | 3 uploads | 4 uploads |
| ---: | ---: | ---: | ---: | ---: |
| 2 (`k=1..2`) | 0.01627 / 0.50 | 2.92e-15 / 1.00 | 5.70e-14 / 1.00 | 5.50e-11 / 1.00 |
| 3 (`k=1..3`) | 0.02246 / 0.33 | 0.007098 / 0.67 | 1.98e-13 / 1.00 | 3.47e-13 / 1.00 |
| 4 (`k=1..4`) | 0.02660 / 0.25 | 0.010627 / 0.50 | 0.003268 / 0.75 | 1.78e-13 / 1.00 |

All ten initialization trials in each of the 12 conditions completed successfully. The one-upload implementation matches the prior model to numerical precision; its largest paired test-MSE difference from the earlier sweep was `3.64e-11`. A separate scan over 24 random parameter settings per upload count found accessible support through `k=L` for `L=1..4`. The trained model recovered all target frequencies in every condition where upload count reached the highest target frequency. This is strong proof-of-concept evidence that **two qubits can fit these four-frequency targets when the input is re-uploaded**.

The targets are nested consecutive harmonics, `k=1..N`. In this experiment, the number of components therefore equals the highest harmonic. The evidence supports upload count tracking the **highest frequency** (the spectrum grows through `k=L` for `L` uploads); it does not establish a general one-upload-per-nonzero-component rule. For example, the bandwidth bound predicts a target containing only `k=4` still needs four uploads, while `{k=1,k=4}` also needs four. A sparse-target comparison is added to the tree to test this directly.

There is one design limitation: upload count and trainable depth increased together (7, 14, 21, or 28 parameters). The random-circuit scan separates the expressivity result from optimization, but a depth-matched one-upload control is still needed to isolate the effect of repeating `x` on trained performance. Only one fabricated dataset was used, so repeat on independent data seeds before making a robustness claim. For multi-upload circuits the unitary depends on `x`, so canonical-QFT process fidelity is not a single suitable comparison metric.

**Reading the MSE plot:** the connected values are mean test MSE across ten initializations; whiskers show ±1 population standard deviation, not the minimum or a boxplot. The sharp drop for the four-frequency target at four uploads is the actual mean reaching about `1.8e-13`, once `k=4` is available. For the two-frequency target, the small rise from `2.9e-15` to `5.5e-11` across extra uploads is still numerical-zero-level error; tiny optimizer residuals vary non-monotonically. On the logarithmic plot, a lower whisker is omitted when `mean - SD <= 0` because zero/negative values cannot be shown on that axis.

The grouped [re-upload report](results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/index.html) includes [MSE versus uploads](results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/test_mse_vs_uploads.png), [frequency-support recall](results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/frequency_support_recall.png), and [learned spectra](results/periodic-function-mse-zero-offset-reupload-sweep-z1-readout/learned_spectra_by_uploads.png), with per-run reports and metrics. The tree now tracks both the matched-depth control and the sparse-target comparison of component count versus maximum harmonic.

The grouped [results page](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/index.html) puts the overall lowest- and highest-MSE runs first and includes each run report. Its summary graphs are outside the per-run folders: [per-gate angle-change histograms](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/parameter_change_histograms.png), [seven-angle initial/final spider plots](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/parameter_spider_profiles.png), and [per-gate circular angle transitions](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/u0_theta_circular_changes.png) (with a separate circle plot for every gate). [Run metrics CSV](results/periodic-function-mse-zero-offset-component-sweep-z1-readout/metrics.csv) records train/validation/test sample counts for every run.

The single-upload study revealed a frequency-capacity limit for that encoding/readout, and the re-upload sweep showed improved task fitting while changing both encoding structure and trainable depth. These results motivate a lab decision among fixed encoding variants, a controlled re-upload comparison, and direct QFT recovery. No one option is designated as the preferred ansatz. Any next function-learning study should retain zero intercept for continuity and report MSE, coefficient error, and frequency-support precision/recall; QFT process fidelity remains a separate outcome.

## Design question for lab discussion

**How should we vary the encoding while keeping the central goal testable?** Candidate comparisons:

- **Fixed encoding variants:** use the original seven-parameter trainable QFT-layout block and compare predetermined encodings (for example, fixed integer scales of `RY(2*pi*x)`). This isolates the encoding choice from trainable block count. It still changes the model's accessible function class.
- **Repeated data encoding:** insert the same fixed `RY(2*pi*x)` encoding between trainable blocks. This expands accessible frequencies, but changes the circuit structure; the prior sweep also increased trainable parameters with upload count. A matched-depth single-upload control is needed to isolate repetition.
- **Direct QFT recovery:** train from randomized parameters toward the known full QFT and assess whether the algorithm itself is recovered, separately from scalar function MSE.

The group should choose a branch, target, matched controls, and success tolerances before the next sweep. The scientific outcome remains whether the circuit recovers QFT or succeeds at the task without doing so, not which architecture wins a benchmark.

## Metric definitions

- **MSE:** average squared prediction-target difference on a named data split.
- **Absolute error over one period:** `e(x)=|f_theta(x)-g(x)|` on a dense grid over `[0,1)`. Reports give its mean and maximum. It is pointwise error, not MSE.
- **Process fidelity:** normalized Frobenius/Hilbert–Schmidt overlap of the learned unitary block and canonical QFT. It is diagnostic only in the MSE experiment.
- **Fourier regression:** classical least squares on a constant plus sine/cosine basis. Here it receives `k=1`, so it shows the fit available when the frequency is supplied, not whether the QNN discovers it.

## Evidence and reports

- [Grouped trial report](results/periodic-function-mse-z1-readout/index.html)
- [Grouped machine-readable results](results/periodic-function-mse-z1-readout/summary.json)
- [Zero-offset multi-dataset report](results/periodic-function-mse-zero-offset-multidataset-z1-readout/index.html)
- [Zero-offset multi-dataset results JSON](results/periodic-function-mse-zero-offset-multidataset-z1-readout/summary.json)
- [LaTeX methodology and results](METHODOLOGY.tex)
- [Machine-readable tree](hypotheses/HYPOTHESIS_TREE.json)

Regenerate per-run plots and reports from saved data without retraining:

```bash
python scripts/refresh_function_mse_reports.py
```
