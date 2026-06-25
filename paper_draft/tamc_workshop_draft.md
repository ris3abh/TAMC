# TAMC: Topological Attractor Meta-Control for Forward-Only Test-Time Adaptation in Non-Stationary Time-Series Forecasting

> **Draft status:** First workshop-style draft, assembled from `paper_notes/` and repo results. Not submission-ready. See `paper_draft/README.md` for scope and next steps. All numeric claims below are traceable to files in `figures/` and `paper_notes/research_brief.md` (Section 19) at the time of writing.

## Abstract

Time-series forecasters deployed in the real world are typically frozen after training, yet the dynamics generating their input stream rarely stay fixed. A shift can change the *shape* of the underlying dynamics — periodic behavior becoming quasi-periodic, a stable equilibrium becoming chaotic, a discrete recurrence crossing a bifurcation — while leaving first-order statistics such as mean, variance, and dominant frequency largely unchanged. Detectors built on those statistics are structurally blind to this class of shift. We study whether persistent homology of a streaming, delay-reconstructed attractor can serve as a causal, forward-only **meta-control** signal — a gate that decides *when* and *how much* a lightweight adapter should act around a frozen forecaster, rather than an additive term in any optimization objective or a fixed-on correction. We call this framework TAMC (Topological Attractor Meta-Control). On three controlled dynamical-system detection tasks (sine-to-quasi-periodic, logistic-map bifurcation, Lorenz stable-equilibrium-to-chaos), TAMC's topological drift signal is competitive with or better than autocorrelation and spectral drift, and clearly ahead of mean/variance drift. In forecast-adaptation experiments, TAMC-gated blending improves the adaptation tradeoff on the original sine task and gives the best result on a Lorenz shift, while reducing harm relative to always-on adaptation on a logistic-map shift where the frozen forecaster remains optimal; with a more modest margin, it also has the best tradeoff on a real series (ETTh1) under a controlled injected perturbation when compared only against autocorrelation-, spectral-, and mean/variance-gated baselines. A stronger rolling local AR forecaster outperforms TAMC on the sine task but is harmful on the logistic-map and Lorenz shifts, showing that topological control and adaptive-forecaster inductive bias both matter; two simpler additive-residual adapters failed to net any benefit, which we report as evidence rather than omit. However, once a direct statistical/distributional regime-similarity gate (an RG-style baseline, conceptually the closest neighbor to TAMC's control mechanism) is added and the comparison is extended to four more real datasets, the real-data picture reverses: TAMC does not beat this simpler regime-similarity gate on any of the five real datasets tested, directly weakening the case for topological drift over statistical alternatives specifically on real data, in contrast to its clear and consistent advantage on the synthetic dynamical-systems detection tasks. A systematic ablation over homology dimension, embedding delay, and window size corrects an earlier, weaker hypothesis: H0 (connected-component structure) is the more robust default across every system and grid cell tested, not H1 (loop structure), despite H1 being the more intuitive choice for loop-like attractors. A runtime benchmark shows persistent homology is the dominant cost of TAMC, three to four orders of magnitude slower per window than the non-topological baselines at the window sizes tested, which bounds where TAMC is currently practical. We present this as an early-stage research prototype: the evidence base is controlled and synthetic-dominated, the adaptive forecaster is a simple heuristic, and direct comparison against learned non-topological TTA baselines (DynaTTA, COSA, PETSA) remains future work.

## 1. Introduction

Time-series forecasters are typically fit on a window of historical data and then deployed to predict a stream that keeps arriving after that window ends. In practice, the dynamics generating that stream rarely stay fixed forever: weather patterns shift, traffic routing changes, financial regimes turn over, sensors degrade, physiological state changes. A forecaster that was accurate on the regime it was fit to can degrade sharply once the underlying dynamics change, even though nothing about the forecaster itself changed.

This motivates a forward-only, label-free test-time adaptation (TTA) setting: at each time `t`, only the observed history `x_{<=t}` is available — there is no ground-truth future value to compute a loss against, and the forecaster's weights are never updated. The central question is:

> How can a frozen or black-box time-series forecaster adapt online to changing dynamics without labels, gradients, or model-weight updates?

A natural first instinct is to monitor simple statistics of the stream — rolling mean, rolling variance, autocorrelation, spectral content — and trigger adaptation when they move. This works well when a shift changes those statistics directly, but many structurally important shifts do not: a periodic signal can become quasi-periodic while keeping the same mean, variance, and dominant frequency; a system can move from a stable equilibrium into chaos while its marginal statistics stay within a similar range; a discrete recurrence can cross a bifurcation threshold into chaotic behavior with first-order statistics that barely move. In each case, what has actually changed is the *shape* of the dynamics — the structure of the attractor the system's trajectory lives on — rather than its first- or second-order statistics.

Takens' delay embedding theorem gives a way to reconstruct the geometry of a dynamical system's attractor from a single observed coordinate: stacking delayed copies of the signal into a point cloud recovers, under mild conditions, a space diffeomorphic to the system's true (possibly unobserved, possibly multi-dimensional) phase-space attractor. Persistent homology summarizes the multi-scale topological structure of that point cloud — connected components (H0), loops (H1), and their persistence across scales — giving a structural fingerprint of the dynamics that is sensitive to exactly the kind of shape change that first-order statistics can miss.

Detecting drift, however, is necessary but not sufficient. A detector that fires correctly but is wired into an adapter naively can still hurt more than it helps: an adapter that is always "on" degrades in-distribution accuracy even when nothing has shifted. TAMC's contribution is to use the topological drift signal as a **meta-control** signal — a gate, rate, or selector that decides *when* and *how much* a forward-only adapter should act — rather than as a label, a loss term, or a fixed-on correction. Concretely, the closed loop is:

```text
stream -> delay embedding -> persistent homology -> drift score -> gate/control signal -> adapted forecast
```

TAMC is not proposed as a generic topological time-series detector, nor as a topology-trained forecasting architecture: prior work has already explored persistent homology for forecasting, streaming topological drift detection, and topology-guided TTA outside time series (Section 8). The contribution we study here is the closed-loop control mechanism — using topological attractor drift to gate a forward-only adapter around a frozen forecaster — not topological drift detection in isolation.

**Contributions.**

1. A causal, forward-only topological drift signal (`src/tamic_signal.py`) built from delay-reconstructed attractor windows and Wasserstein distance between persistence diagrams, used strictly as a multiplicative gate rather than an additive loss term.
2. TAMC-Lite forecast blending (`src/tamc_pipeline.py::TamicBlendPipeline`), which gates a blend between a frozen forecaster and a forward-only adaptive forecaster, evaluated against two weaker additive-residual adapter designs and across sine, logistic-map, Lorenz, and ETTh1 controlled perturbations. The results show both positive cases and failure modes: topology-gated blending can improve the adaptation tradeoff and reduce harmful adaptation relative to always-on blending, but its net value depends on the adaptive forecaster being gated having the right inductive bias for the shift.
3. Three controlled dynamical-system detection benchmarks, one real-data controlled-perturbation benchmark, and forecast-adaptation benchmarks on all four of those settings, with explicit adaptation tradeoff metrics that separate the cost of false adaptation from the benefit of true adaptation.
4. A systematic ablation over homology dimension, delay, and window size, and a runtime benchmark quantifying TAMC's computational cost — both reported honestly, including a correction to an earlier, weaker hypothesis about which homology dimension is best.

## 2. Problem Setup

Let `x_1, x_2, ..., x_t, ...` be a univariate stream observed sequentially, generated by an underlying dynamical system whose governing parameters may change at unknown, unlabeled times. We call each interval during which the governing dynamics stay fixed a **regime**.

A **frozen forecaster**

```text
f_theta: R^L -> R^H
y_hat_t = f_theta(x_{t-L:t})
```

maps a context window of length `L` to a forecast of horizon `H`. `theta` is fixed at deployment time and never updated after `f_theta` is fit once, on data drawn only from the **source regime**, before any shift occurs.

**Strict causal information constraint.** At every time `t`, any computation — the forecaster's prediction, the topological drift score, the adapter's gate or correction — uses only information available causally: `x_1, ..., x_t`. Future values `x_{t+1}, x_{t+2}, ...` and any ground truth for forecasts already made are unavailable. This rules out two common sources of leakage in time-series TTA evaluation: (1) future-label leakage, using `x_{t+1:t+H}` to compute or calibrate anything at time `t`; and (2) whole-series statistics leakage, computing a normalization, threshold, or source prototype from data spanning both regimes rather than only the source regime or the causal past. Every experiment in this work enforces this by construction — source prototypes and frozen-forecaster fits are built only from the source segment, and all rolling computations index strictly backward from the current step.

**Source regime, shifted regime, context length, horizon.** The **source regime** is the initial segment of the stream; `f_theta` and the topological source prototype `D_source` (Section 3.2) are built only from it. The **shifted regime** is the segment after the shift; no data from it is used to fit `f_theta` or `D_source`. **Context length** (`L`) is the number of most-recent observations fed to a forecaster at prediction time. **Forecast horizon** (`H`) is the number of future steps predicted in one call.

**What "forward-only" means here.** Two senses apply simultaneously: (1) no weight updates — `f_theta`'s parameters never change after fitting; any adaptation happens in output space (a gate, a blend, a residual correction) or by running a second, separately defined forecaster; (2) no gradients through an adapter's own decision when its small parameter vector is searched (relevant to `ForwardOnlyOptimizer` in `src/adapters.py`, not exercised in the reported results below) — any such search would use only forward evaluations of a fitness function. Both senses share the same motivating constraint: TAMC should work for any frozen forecaster treated as a black box, including ones too large, too expensive, or too inaccessible to fine-tune at test time.

Given this setup, the open question is *how* to decide when and how much output-space adaptation should occur, using only the causal past and no labels. TAMC's proposal is to compute a topological drift score from the delay-reconstructed attractor of the causal past and use it strictly as a meta-control signal over a forward-only adapter.

## 3. Method

[Figure 1: TAMC pipeline schematic — see figures/paper_pipeline_schematic.png]

### 3.1 Delay-Reconstructed Attractor Windows

For a window of raw signal values of length `window` ending at time `t`, TAMC builds a delay-coordinate (Takens) embedding with parameters `(dimension, delay)`: each point in the resulting point cloud is `[x_i, x_{i+delay}, ..., x_{i+(dimension-1)*delay}]`. This reconstructs the qualitative shape of the system's attractor — its loops, lobes, and clustering structure — from a single observed coordinate, even when the true generating system is higher-dimensional and only partially observed (e.g. the `x` component of a 3D Lorenz system).

### 3.2 Persistent-Homology Drift Score

The embedded point cloud's Vietoris-Rips persistence diagram is computed up to a chosen maximum homology dimension, yielding birth-death pairs for connected components (H0) and loops (H1). A single reference window from the source regime defines a stored source diagram `D_source`, built once, offline. For the diagram `D_t` of the current window (at one chosen homology dimension, `drift_dimension`), TAMC scores drift as the Wasserstein distance:

```text
delta_t = W(D_t, D_source)
```

`delta_t` is purely a function of the observed window; it never depends on which adapter or candidate forecast is being evaluated. A monitor keeps a running history of `delta_t` and z-scores new observations against that history; before enough history has accumulated, the z-score is defined as `0` (no gate action) rather than an unstable early estimate.

### 3.3 Topological Meta-Control

A drift score alone says only how different the current window is from the source regime, not what correction to apply, and using it as an additive term in a forward-only/zeroth-order optimization objective is methodologically broken whenever the term does not depend on the candidate being ranked (it becomes a constant offset that contributes nothing to ranking). TAMC instead uses `delta_t` only as a **control signal** — multiplying or selecting among forecasts/corrections:

```text
g_t    = sigmoid(z(delta_t) - threshold)                        # gate in [0, 1]
eta_t  = eta_min + (eta_max - eta_min) * sigmoid(z(delta_t))     # update rate (not used in reported results)
k_star = argmin_k distance(D_t, D_source_k)                      # prototype selection (not used in reported results)
```

Every adapter exercised in the results below follows this rule: the gate only ever multiplies a correction or blends two existing forecasts; it is never added to an optimization objective.

To compare TAMC against non-topological drift signals fairly, the same `history -> z(distance) -> sigmoid(z - threshold)` control law is applied to mean/variance, autocorrelation, and spectral drift scores (`src/drift_gates.py::ScalarDriftSignal`), decoupled from persistent homology. This isolates *which drift signal is the better control input* from *whose gate formula is implemented better* — without it, a result favoring TAMC could always be explained away as an artifact of how its gate happens to be shaped.

### 3.4 TAMC-Lite Forecast Blending

The strongest current adaptation result (Section 5.2) comes from gating a blend between two full forecasts rather than an additive residual correction. Two forecasters are evaluated on the same context at every step: `y_frozen`, fit once on the source regime and never updated; and `y_adaptive`, a forward-only forecaster that only ever sees the current context (no labels, no future data, no gradient updates). The topological gate blends them:

```text
y_blend_t = (1 - g_t) * y_frozen_t + g_t * y_adaptive_t
```

When the gate is near `0` (no detected drift), `y_blend_t ~= y_frozen_t`, preserving the frozen forecaster's accuracy on the regime it was fit to. When the gate is near `1` (drift detected), `y_blend_t` shifts toward the adaptive forecaster, which has no stake in the old regime. This is in deliberate contrast to two earlier, weaker adapter designs (`MeanShiftResidual`, `AnalogResidualAdapter`) that add a residual correction to the frozen forecast rather than blending two full forecasts; both are reported in Section 5.2 because they underperformed the blend, not omitted because they did.

## 4. Experimental Setup

### 4.1 Detection Tasks

Three controlled, synthetic dynamical-systems experiments test whether topological drift detection generalizes across qualitatively different kinds of regime shift, each with a known, controlled shift boundary and matched mean/variance between regimes where applicable:

- **Sine to quasi-periodic** (`experiments/synthetic_regime_shift.py`): a clean periodic loop shifts to the same signal plus a second incommensurate frequency.
- **Logistic map, hardened** (`experiments/logistic_map_shift.py`): `r = 3.45` (periodic-ish) shifts to `r = 3.75` (chaotic), with observation noise added so the comparison is not trivial (an earlier, harder-edge parameterization gave every method, including naive mean/variance drift, a trivial perfect score).
- **Lorenz** (`experiments/lorenz_shift.py`): a stable equilibrium (`rho = 20`) shifts to the classic chaotic butterfly attractor (`rho = 28`); only the `x` coordinate is observed and reconstructed via delay embedding.

### 4.2 Forecast Adaptation Tasks

- **Synthetic** (`experiments/tamc_lite_synthetic_forecast.py`): the sine-to-quasi-periodic shift, with a `LinearARForecaster` (ridge-regularized linear autoregression) fit only on the source regime as the frozen forecaster, and `RecentPatternForecaster` (autocorrelation-lag-based seasonal-naive-with-drift) as the forward-only adaptive forecaster.
- **Real-data controlled perturbation** (`experiments/real_data_controlled_shift.py`): the ETTh1 dataset's `OT` column, loaded from a local CSV file (never downloaded by the script), causally normalized using only the source half. A deterministic perturbation is injected into the second half only; the main reported result uses `seasonality_break` (mixing the post-shift segment with its own reverse-time copy, inverting the phase of any recurrence structure), with a separate stochastic `noise` shift type used as a robustness check.

### 4.3 Baselines

- **Detection:** rolling mean drift, rolling variance drift, autocorrelation-signature L2 distance, spectral-signature (FFT power spectrum) L2 distance.
- **Forecast adaptation:** the frozen forecaster alone; the adaptive forecaster alone; an always-on 50/50 blend (no gate); `MeanShiftResidual` (residual correction from recent-context mean vs. source mean) and `AnalogResidualAdapter` (k-nearest-neighbor lookup over source-regime context/residual pairs), each both always-on and TAMC-gated; and, on the real-data task, the same blend gated by mean/variance, autocorrelation, and spectral drift under the identical `ScalarDriftSignal` control law described in Section 3.3.

### 4.4 Metrics

**Detection:** AUROC (treating post-shift windows as the positive class against the drift score), detection delay (steps from the true shift boundary to the first post-shift window exceeding a `pre_mean + n_std * pre_std` threshold, `n_std = 3` throughout), false alarms (pre-shift windows exceeding that threshold), and separation (`(post_mean - pre_mean) / pre_std`).

**Forecast adaptation:** MAE and RMSE, computed separately on pre-shift and post-shift windows, plus four tradeoff metrics relative to the frozen forecaster baseline:

```text
pre_harm             = MAE_pre_variant  - MAE_pre_frozen
post_gain            = MAE_post_frozen  - MAE_post_variant
net_adaptation_score = post_gain - pre_harm
post_improvement_pct = 100 * post_gain / MAE_post_frozen
```

`net_adaptation_score` nets the cost of adapting when nothing has shifted against the benefit of adapting when something has; it is the metric we lead with. `post_improvement_pct` is reported for interpretability but is a per-seed ratio with a denominator that varies substantially across seeds, so its mean/std can diverge sharply from what the underlying MAE numbers suggest — we report both numbers honestly rather than only the more flattering one (Section 5.2). For the frozen forecaster itself, all four tradeoff metrics are exactly `0` by construction.

## 5. Results

### 5.1 Controlled Dynamical-System Detection

[Figure 2: Detection AUROC summary — see figures/paper_detection_summary.png]

10-seed results (mean +/- std where reported; full tables in `figures/synthetic_regime_shift_multiseed_metrics.csv`, `figures/logistic_map_shift_multiseed_metrics.csv`, `figures/lorenz_shift_multiseed_metrics.csv`):

| System | Method | AUROC | Delay (steps) | False Alarms | Separation |
|---|---|---|---|---|---|
| Sine -> quasi-periodic | **TAMC (H1)** | **0.9963 +/- 0.0038** | **11.0** | 0.0 | 136.2 |
| | Autocorrelation | 0.9728 | 91.0 | 0.0 | 5.0 |
| | Spectral | 0.9320 | 113.4 | 0.0 | 2.4 |
| | Rolling variance | 0.6410 | 11.0 | 0.0 | 0.9 |
| | Rolling mean | 0.5112 | n/a | 0.0 | 0.02 |
| Logistic map (hardened) | **TAMC (H0)** | **0.9998 +/- 0.0007** | **3.8** | 0.0 | 43.0 |
| | Spectral | 0.9997 | 3.8 | 0.0 | 2868.1 |
| | Autocorrelation | 0.9989 | 6.2 | 0.6 | 428.3 |
| | Rolling mean | 0.7566 | 79.0 | 0.0 | 2.5 |
| | Rolling variance | 0.7214 | 115.0 | 0.0 | 2.3 |
| Lorenz (equilibrium -> chaos) | **TAMC (H0)** | **0.9635 +/- 0.0431** | **70.6** | 0.0 | 6.3 |
| | Spectral | 0.9489 | 117.0 | 0.0 | 4.6 |
| | Autocorrelation | 0.9419 | 113.0 | 0.0 | 4.7 |
| | Rolling mean | 0.8793 | 61.4 | 0.0 | 5.7 |
| | Rolling variance | 0.6259 | 186.0 | 0.0 | 1.4 |

These are the original per-system detection runs, not the ablation-selected best configurations; the later ablation (Section 6.1) shows that H0 is the more robust default across the full tested delay/window grid, including on the sine system reported here with H1.

TAMC's topological drift is the best or tied-for-best AUROC on every system, has zero false alarms throughout, and on Lorenz beats every baseline on every reported axis (AUROC, delay, separation). On the logistic map, TAMC, spectral, and autocorrelation are all near-ceiling on AUROC (>= 0.999), with TAMC and spectral tied on delay; spectral's much larger separation value reflects scale, not better discrimination (AUROC and delay are the metrics we treat as primary). On all three systems, naive mean/variance drift is clearly the weakest method.

### 5.2 TAMC-Lite Forecast Adaptation

[Figure 3: Forecast adaptation tradeoff summary — see figures/paper_adaptation_tradeoff.png]

10-seed results on the synthetic sine-to-quasi-periodic forecast task (`figures/tamc_lite_synthetic_forecast_multiseed_metrics.csv`):

| Variant | Pre-shift MAE | Post-shift MAE | Pre Harm | Post Gain | Net Adaptation Score |
|---|---|---|---|---|---|
| Frozen forecaster | 0.0218 | 0.4448 | 0.0000 | 0.0000 | 0.0000 |
| Adaptive recent-pattern alone | 0.0775 | 0.6228 | 0.0558 | -0.1779 | -0.2337 |
| Always-on 50/50 blend | 0.0435 | 0.4113 | 0.0217 | 0.0336 | 0.0119 |
| Always-on MeanShiftResidual | 0.1115 | 0.4743 | 0.0897 | -0.0295 | -0.1192 |
| Always-on AnalogResidualAdapter | 0.0175 | 0.4450 | -0.0043 | -0.0001 | 0.0041 |
| TAMC-gated MeanShiftResidual | 0.0346 | 0.4568 | 0.0128 | -0.0120 | -0.0248 |
| TAMC-gated AnalogResidualAdapter | 0.0205 | 0.4449 | -0.0013 | -0.0001 | 0.0012 |
| **TAMC-gated blend** | **0.0297** | **0.4112** | **0.0080** | **0.0336** | **0.0257** |

The TAMC-gated blend has the best Net Adaptation Score of every variant tested: it matches the always-on blend's post-shift gain (~0.034 either way) while incurring much less pre-shift harm (0.008 vs. 0.022). The two additive residual adapters are reported as a genuine negative result, not omitted: `MeanShiftResidual` actively hurts pre-shift accuracy (it reacts to within-cycle phase, not real drift) and never recovers post-shift, regardless of gating; `AnalogResidualAdapter` is statistically indistinguishable from the frozen forecaster everywhere, because it has no capacity to express a correction larger than the (tiny) in-sample fitting residual it memorizes from the source regime. Adapting via residual correction failed on this task; adapting via gated blending between two full forecasts worked.

#### Extension to logistic-map and Lorenz forecast adaptation

[Figure 6: Dynamical forecast adaptation summary — see figures/paper_dynamical_adaptation_summary.png]

`experiments/dynamical_forecast_adaptation.py` repeats the comparison on `logistic_map` and `lorenz` (same causal generators as Section 5.1), adding a second standalone adaptive forecaster, `RollingLinearARForecaster` (refits a small ridge-regularized AR model from the current context window alone at every prediction call), and gating both adaptive forecasters with TAMC as well as with mean/variance-, autocorrelation-, and spectral-drift under the identical control law from Section 3.3. 10-seed results (full table in `figures/dynamical_forecast_adaptation_tradeoff_summary.csv`):

| System | TAMC-gated recent-pattern | Always-on recent-pattern | Best overall | Interpretation |
|---|---:|---:|---|---|
| Sine -> quasi-periodic | +0.0335 | +0.0119 | RollingLinearAR alone (+0.2673) | TAMC helps vs. always-on, but rolling AR is better |
| Logistic map | -0.2855 | -0.5211 | Frozen forecaster (0.0000) | no adapter helps; TAMC reduces harm |
| Lorenz | +0.1668 | +0.1462 | TAMC-gated blend (+0.1668) | genuine TAMC adaptation win |

The result is mixed, and we report it as such rather than as a clean generalization. On **Lorenz**, TAMC-gated blending is the best variant overall, beating always-on blending and all three non-topological gates (which range from -0.61 to -0.71) — a genuine new win beyond the sine-only evidence above. On **logistic map**, nothing helps: every adaptive or blended variant underperforms the frozen forecaster alone, including TAMC-gated blending; TAMC's gate is still the *least harmful* non-frozen option (-0.29 vs. always-on's -0.52 and the non-topological gates' -1.43 to -1.44), so it correctly limits damage even though it cannot manufacture a benefit the underlying forecasters do not have. On **sine**, TAMC-gated blending remains positive but is no longer the best choice once `RollingLinearARForecaster` is added to the comparison: a standalone, ungated rolling-AR forecaster, and even autocorrelation-gated blending paired with that same forecaster, both outperform TAMC-gated blending here. `RollingLinearARForecaster` is also reliably harmful as a *standalone* forecaster on logistic map and Lorenz (Net Adaptation Score -2.5 and -4.7 respectively), so it is not a safe drop-in replacement for `RecentPatternForecaster` despite occasionally outperforming it once blended.

These results remove the earlier limitation that forecast adaptation had only been tested on the sine task, but they do not establish universal adaptation gains. They show that topological gating is useful when the gated adaptive forecaster has the right inductive bias; when the adapter is mismatched, topology can reduce harm but not guarantee improvement over the frozen forecaster. We read this as evidence that topology-gated adaptation generalizes to *some* but not all of the systems where TAMC detection already works, not as evidence that it generalizes uniformly.

### 5.3 Real-Data Controlled Perturbations

Dataset: ETTh1, `OT` column, loaded from a local CSV file. 10-seed Net Adaptation Score under a controlled, injected `seasonality_break` perturbation (`figures/real_data_controlled_shift_seasonality_break_multiseed_metrics.csv`):

| Gate | Net Adaptation Score |
|---|---|
| **TAMC-gated blend** | **0.0027** |
| Autocorrelation-gated | 0.0015 |
| Spectral-gated | -0.0003 |
| Mean/variance-gated | -0.0052 |
| Always-on 50/50 blend | -0.0305 |
| Adaptive recent-pattern alone | -0.1520 |

TAMC has the best adaptation tradeoff of every gate tested on this real series, including the three non-topological gates compared under the identical control law described in Section 3.3 — but the margin is modest (0.0027 vs. autocorrelation's 0.0015), not the much clearer separation seen in the synthetic detection experiments. Because `seasonality_break` (like three of the other four supported shift types) is a fully deterministic perturbation given the input segment, multi-seed runs on it correctly show zero standard deviation across seeds; this is expected behavior, not evidence of robustness by itself.

**Stochastic robustness check.** The `noise` shift type is the only one that actually varies across seeds, making it the honest test of whether the result above is robust or a lucky deterministic draw:

| Gate | Net Adaptation Score (mean +/- std) |
|---|---|
| **TAMC-gated blend** | **0.0011 +/- 0.0015** |
| Autocorrelation-gated | -0.0015 +/- 0.0017 |
| Spectral-gated | -0.0018 +/- 0.0017 |
| Mean/variance-gated | -0.0034 +/- 0.0021 |
| Always-on 50/50 blend | -0.0394 +/- 0.0060 |
| Adaptive recent-pattern alone | -0.1853 +/- 0.0152 |

TAMC-gated blend has the highest mean and is the only gate with a positive mean Net Adaptation Score under pure noise. This should not be oversold: its mean (0.0011) is smaller than its own standard deviation (0.0015), so the result is not statistically distinguishable from zero. We read this honestly as TAMC being the least-bad / most net-neutral option under pure noise, not a clear winner — when the injected "shift" has no real structure for any drift signal (topological or not) to detect, no gate tested provides a robust net benefit. This suggests topology is more useful for *structural/dynamical* shifts than as a generic noise-robustness mechanism, consistent with the motivation in Section 1.

Both real-data results are controlled, injected perturbations on a real series, not naturally occurring distribution shift; they establish that the method still functions on real (non-synthetic) values and noise characteristics, not that it would detect or usefully gate adaptation to a real-world regime change arising on its own.

#### Multi-dataset benchmark against an RG-style regime-similarity gate

[Figure 7: Real-data benchmark, TAMC vs. RG-style regime-similarity gate — see figures/paper_regime_benchmark_summary.png]

The ETTh1-only result above predates a direct comparison against the closest conceptual neighbor to TAMC's control mechanism: RG-TTA (Regime-Guided Test-Time Adaptation), which uses distributional regime similarity rather than topology to control TTA intensity. `experiments/benchmark_regime_control.py` adds this comparison -- a transparent, from-scratch RG-style statistical/distributional regime-similarity gate (`src/regime_similarity.py`; no RG-TTA code used), gated under the identical control law as every other gate -- across five real datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather) under the same `seasonality_break` injection protocol. 10-seed results (`seasonality_break` is fully deterministic, so all seeds are numerically identical; full table in `figures/benchmark_regime_control_tradeoff_summary.csv`):

| Dataset | TAMC NAS (rank /8) | RG-style NAS (rank /8) | Best variant |
|---|---:|---:|---|
| ETTh1 | -0.0016 (5th) | -0.0012 (4th) | Autocorrelation-gated (+0.0015) |
| ETTh2 | -0.0102 (7th) | -0.0059 (5th) | Autocorrelation-gated (+0.0061) |
| ETTm1 | +0.0213 (4th) | +0.0245 (3rd) | Always-on blend (+0.0361) |
| ETTm2 | -0.0023 (5th) | +0.0028 (1st) | RG-style-gated (+0.0028) |
| Weather | +0.0010 (4th) | +0.0021 (3rd) | Spectral-gated (+0.0028) |

This is not mixed evidence: **TAMC does not beat the RG-style gate on any of the five datasets**, and never ranks better than 4th of the 8 variants tested. The gap is small on some datasets (ETTm1: 0.0213 vs. 0.0245) and clearer on others (ETTh2, where TAMC is the worst gated blend tested; ETTm2, where RG-style is the single best variant of all 8 and TAMC is net-negative). Autocorrelation-gating, not TAMC, is the best or most competitive non-RG gate on most datasets. Most absolute margins remain small (|NAS| < 0.01 on 4 of 5 datasets), so this is not catastrophic, but it is a consistent directional signal that a simple statistical/distributional regime-similarity gate is currently a stronger real-data control input than TAMC's topological drift gate, under this benchmark's fixed protocol and parameters. We read this as a direct, honest weakening of the case for topology over simpler statistical alternatives on real data specifically -- in contrast to the synthetic dynamical-systems detection results (Section 5.1), where TAMC's advantage over non-topological baselines was clear and consistent.

**Is this specific to `seasonality_break`, or general?** A follow-up diagnostic (`experiments/benchmark_shift_types.py`, 3 seeds) sweeps all five controlled perturbation types (`amplitude`, `trend`, `noise`, `seasonality_break`, `frequency_proxy`) across the same five datasets -- 25 dataset/shift-type cells, 75 runs, all completing without failure. The answer is general, not specific: RG-style beats TAMC outright on `amplitude` (5/5 cells), `trend` (3/5), and `seasonality_break` (4/5); TAMC's only nominal advantages are on `noise` (2 wins, 1 loss, 2 ties) and the more balanced `frequency_proxy` (2 wins, 2 losses, 1 tie). Across all 25 cells, **TAMC never ranks 1st overall** of the 8 variants tested, and beats the RG-style gate outright in only 4 of 25 cells versus 15 RG-style wins. Even the apparent `noise` advantage does not hold up to scrutiny: it is driven almost entirely by one outlier cell (ETTm2), where the actual best variant is the *ungated* adaptive forecaster, not the TAMC-gated blend. We read this as confirmation that the real-data weakness is a property of the method under this protocol, not an artifact of one perturbation type.

## 6. Ablations

### 6.1 Homology Dimension, Delay, and Window Size

[Figure 4: Topology ablation heatmap — see figures/paper_ablation_heatmap.png]

The detection results in Section 5.1 each used one homology dimension hand-picked per system (H1 for sine, H0 for logistic map and Lorenz), motivated post hoc by which dimension seemed to fit each system's attractor shape. `experiments/topology_ablation.py` checks this systematically: 10 seeds x 3 systems x 2 homology dimensions (H0, H1) x 5 delays (2, 4, 6, 8, 12) x 2 window sizes (64, 128) = 600 runs, with embedding dimension fixed at 3 and no system-specific tuning introduced for the ablation. The grid ran to completion in 1227.8 seconds (~20.5 minutes); full results in `figures/topology_ablation_summary.csv`.

| System | H0 AUROC range (10-seed mean) | H1 AUROC range (10-seed mean) |
|---|---|---|
| Sine -> quasi-periodic | 0.9945 - 0.9984 (std <= 0.005) | 0.857 - 0.996 (one cell collapses) |
| Logistic map | 0.9991 - 0.9998 (std <= 0.0025) | 0.984 - 0.996 (std up to 0.013) |
| Lorenz | 0.913 - 0.982 (std up to 0.13) | 0.516 - 0.801 (std 0.20 - 0.33 throughout) |

**This corrects the earlier, single-data-point hypothesis.** The systematic sweep does not support "H1 is strongest for loop-like periodic/quasi-periodic attractors": H0 is uniformly excellent and low-variance across the *entire* grid for all three systems, including the two loop-like ones. H1 is competitive with H0 on most of the sine/logistic-map grid (occasionally a hair ahead on an individual cell) but is never clearly better, and has a real failure mode H0 does not share — at `delay=12, window=64`, sine's H1 AUROC collapses to 0.857 +/- 0.031 while H0 stays at 0.994 +/- 0.004 in that same cell. The Lorenz half of the original hypothesis holds strongly: H0 clearly and consistently beats a much weaker, far more seed-unstable H1. The revised, ablation-supported takeaway is that **H0 is the more robust default across every system and grid cell tested; H1 is intuitively motivated for loop-like attractors and can match H0 there, but is never clearly better and is unreliable on Lorenz's compact-to-chaotic transition.** We report this as a correction rather than retrofitting the earlier claim, because we believe an ablation, not a single-run anecdote, is the right basis for asserting which modeling choice "wins."

## 7. Runtime Analysis

[Figure 5: Runtime benchmark — see figures/paper_runtime_summary.png]

`experiments/runtime_benchmark.py` measures per-window wall-clock cost (`time.perf_counter()`, 3 repeats, mean reported) for TAMC's H0 and H1 drift against the three non-topological baselines, on the sine-to-quasi-periodic series, at window sizes 64/128/192 (stride 8, 100 windows, embedding dimension 3, delay 8). Persistence is computed once per window for both H0 and H1 (not duplicated), but each method's reported cost is its full standalone cost — matching how the drift signal is actually used elsewhere in this work, with one fixed homology dimension and no cost-sharing across dimensions — not an artificially halved "shared" figure.

| Window | TAMC H0 (sec/window) | TAMC H1 (sec/window) | Spectral (sec/window) | Autocorrelation (sec/window) | H0 vs. spectral | H0 vs. autocorrelation |
|---|---|---|---|---|---|---|
| 64 | 0.0049 | 0.0029 | 0.000013 | 0.000075 | 367.6x | 64.3x |
| 128 | 0.0413 | 0.0396 | 0.000014 | 0.000077 | 3045.1x | 536.6x |
| 192 | 0.1810 | 0.1746 | 0.000014 | 0.000078 | 13178.7x | 2320.0x |

The non-topological baselines are essentially flat with window size, dominated by fixed Python/NumPy overhead rather than the underlying O(window) work. TAMC's cost grows steeply — roughly an order of magnitude per ~1.5x increase in window size — because Vietoris-Rips persistent homology scales poorly with point-cloud size; this is the real, dominant computational cost of TAMC, not an artifact of the benchmark. At window=192, one window of TAMC H0 drift (~181ms) is roughly 13,000x slower than spectral drift (~14 microseconds). At the window sizes actually used in the detection and adaptation experiments above (<=128, stride 8), absolute cost remains in the tens-of-milliseconds range and the experiments above were practical to run; this would not scale to windows in the many hundreds, or to per-step (stride=1) scoring, without mitigation such as sparser striding, smaller windows, approximate/edge-collapse persistence, or scoring only every `k` steps.

## 8. Related Work

We group related work by category rather than provide exhaustive per-paper prose; see `paper_notes/related_work.md` and `paper_notes/research_brief.md` (Section 5) for the full annotated map with links.

- **Forward-only / zeroth-order test-time adaptation** (e.g. FOA, FOZO, EVA-0): adapt models using only forward passes, typically via statistic discrepancy or prediction entropy as the fitness signal. TAMC differs by using topological attractor drift as the meta-control signal instead.
- **Time-series test-time adaptation** (e.g. TAFAS, PETSA, COSA, DynaTTA): TTA frameworks for forecasting using gated calibration, parameter-efficient adapters, output-space correction, or embedding/error-based shift estimation. DynaTTA (real-time shift estimation plus a controlled-shift benchmark) and COSA (lightweight output-space adapter) are the closest direct neighbors to TAMC-Lite's design; TAMC has not yet been benchmarked directly against either (Section 9).
- **Topology for time-series forecasting** (e.g. Topological Attention for Time Series Forecasting, SToPS, N-BEATS + FastZigzag persistence, PH-hybrid forecasters, LISA): use persistent homology (or, for LISA, spectral/diffusion coordinates) as a training-time feature or embedding-parameter-selection tool inside a trainable forecaster. TAMC computes topology at test time as a control signal for a *frozen* forecaster, not as a training-time feature.
- **Topology for streaming data and drift detection** (e.g. streaming persistent homology, persistent-entropy landscape-shift detection, `topoflow`): compute persistent homology over streams or sliding-window delay embeddings to detect regime change. `topoflow` in particular is structurally close to TAMC's detection signal alone; TAMC's distinguishing claim is the adaptation/control loop on top of detection, not detection in isolation.
- **Topology-guided TTA outside time-series forecasting** (e.g. PH over neural activation manifolds, TopoTTA for anomaly segmentation): use persistent homology computed from neural activations or anomaly maps to guide TTA in vision-like settings. TAMC computes topology from the reconstructed data-generating dynamics, not from neural activation manifolds.

**Novelty boundaries.** We do not claim to be first to use persistent homology for time-series forecasting, first to combine delay embeddings with persistent homology, first to use topology for test-time adaptation, or first to compute persistent homology on streaming data — each of these is already established by the lines above. No single line of prior work we are aware of combines all of: persistent homology, computed from delay-reconstructed attractors, on streaming data, used causally to control, forward-only adaptation, of a frozen time-series forecaster. The contribution we study is that specific closed-loop control mechanism.

## 9. Limitations

- **Controlled, synthetic-dominated evidence base.** All forecast-adaptation results except ETTh1 are fully synthetic; the one real-data result uses a controlled, injected perturbation, not a naturally occurring shift.
- **Forecast adaptation gains are conditional, not universal.** Adaptation has now been tested on sine, logistic-map, Lorenz, and ETTh1 controlled perturbations (Section 5.2), but the gains depend on the system: TAMC-gated blending wins clearly on Lorenz and improves safety relative to always-on adaptation on logistic map, but it does not universally beat the frozen forecaster (logistic map) or a stronger local adaptive forecaster (sine, where `RollingLinearARForecaster` alone is better). This suggests topology is a control signal, not a complete adapter: it decides when to trust an adaptive component, but that component must still have an appropriate inductive bias for the shift, and topology cannot supply that bias by itself.
- **TAMC does not beat a simple statistical/distributional regime-similarity gate on real data, and this is general, not specific to one perturbation type.** The ETTh1-only result (modest TAMC lead, 0.0027 vs. autocorrelation's 0.0015) does not hold once a direct RG-style regime-similarity baseline and four more real datasets are added (Section 5.3): TAMC loses to the RG-style gate on all 5 datasets tested under `seasonality_break`, never ranking better than 4th of 8 variants. A follow-up diagnostic sweeping all 5 supported perturbation types confirms this is not an artifact of `seasonality_break`: across 25 dataset/shift-type cells (3 seeds each), TAMC never ranks 1st overall, beats the RG-style gate outright in only 4/25 cells (vs. 15/25 RG-style wins), and its only nominal per-shift-type advantages (`noise`, `frequency_proxy`) are mixed or driven by a single outlier cell rather than a broad effect. This is the most direct and now most thoroughly checked evidence against topology being uniformly superior to simpler statistical alternatives on real data, in contrast to the clear, consistent advantage TAMC's topological drift shows over non-topological baselines on the synthetic detection tasks (Section 5.1).
- **The forward-only adaptive forecasters are simple heuristics**, not learned models: `RecentPatternForecaster` is an autocorrelation-lag-based seasonal-naive-with-drift rule, and `RollingLinearARForecaster` (added when extending to logistic map/Lorenz) is itself unstable as a standalone forecaster on both of those systems (Net Adaptation Score -2.5 and -4.7 respectively).
- **Topological drift is computationally expensive.** Section 7 shows three-to-four-orders-of-magnitude slowdowns versus non-topological drift at the window sizes tested, growing steeply with window size.
- **Homology dimension and embedding parameters matter and are not free to ignore.** Section 6.1's ablation shows H1's effectiveness is not uniform; a different earlier hypothesis (H1 best for loop-like systems) had to be corrected after running the full grid rather than relying on per-system hand-picked choices.
- **The pure-noise adaptation result is weak and not statistically significant.** Under pure Gaussian noise (Section 5.3), TAMC's mean Net Adaptation Score is smaller than its own standard deviation; topology should not be read as a generic noise-robustness mechanism.
- **No direct implementation of, or comparison against, DynaTTA, COSA, or PETSA.** A direct RG-style statistical/distributional regime-similarity gate is now implemented and benchmarked (Section 5.3) -- and currently beats TAMC -- but the other non-topological gates compared (mean/variance, autocorrelation, spectral) remain hand-rolled drift scores under TAMC's own control law, not DynaTTA's/COSA's/PETSA's actual published methods.
- **No foundation-model backbone yet.** All frozen forecasters used here (`LinearARForecaster`, `NaiveLastValueForecaster`) are small, classical models; TimesFM/Chronos/MOMENT/Timer-scale backbones are noted as later-stage candidates in `paper_notes/related_work.md` but not yet exercised.

### Current Missing Experiments

As a concrete, checkable list of what has *not* been run yet, separate from the framing limitations above:

- No naturally occurring real-world regime shift has been tested — only synthetic shifts and one real series (ETTh1) under a controlled, injected perturbation.
- No direct implementation of, or benchmark against, DynaTTA, COSA, or PETSA (Section 8) — only hand-rolled non-topological gates (mean/variance, autocorrelation, spectral) under TAMC's own control law.
- No learned or foundation-model forecaster backbone (TimesFM, Chronos, MOMENT, Timer) — only classical/linear frozen forecasters (`LinearARForecaster`, `NaiveLastValueForecaster`).
- TAMC-Lite forecast blending has now been run on all three detection systems (Section 5.2), with a mixed result (win on Lorenz, no help on logistic map, not the best choice on sine once `RollingLinearARForecaster` is added). What is still missing: a *stronger* forward-only adaptive forecaster that is not itself unstable as a standalone model on logistic map and Lorenz, and a validation-based (rather than fixed-by-decision) way of choosing the adaptive forecaster paired with each non-topological gate.

## 10. Conclusion

We presented TAMC, a framework that uses persistent homology of a streaming, delay-reconstructed attractor as a causal, forward-only meta-control signal — a gate, not a label or loss term — for adapting frozen time-series forecasters under non-stationary dynamics. Across three controlled dynamical-system detection tasks, TAMC's topological drift signal is competitive with or better than autocorrelation and spectral drift and clearly ahead of mean/variance drift. In forecast adaptation, gating a blend between a frozen and a forward-only adaptive forecaster gives the best result on a Lorenz shift, improves the adaptation tradeoff over always-on blending on the original sine task, reduces harm relative to always-on adaptation on a logistic-map shift where the frozen forecaster remains optimal, and — against a narrower set of baselines — has the best tradeoff on one real series under a controlled perturbation, while two simpler additive-residual adapters failed outright and a stronger rolling local AR forecaster outperformed TAMC on sine (though it was harmful as a standalone forecaster on logistic map and Lorenz). That real-data picture changes once a direct statistical/distributional regime-similarity gate (an RG-style baseline, the closest conceptual neighbor to TAMC's control mechanism) is benchmarked across five real datasets: TAMC does not beat this simpler gate on any of them, a clear and consistent finding that weakens the case for topological drift over statistical alternatives specifically on real data, even as TAMC's advantage on the synthetic dynamical-systems detection tasks remains clear. TAMC is best understood as a structural meta-control mechanism, not a universally superior drift signal: it can improve adaptation when the adaptive forecaster it gates is well-suited to the shift and the underlying drift is well-detected by topology, and can reduce damage when adaptation would otherwise be harmful, but on the real datasets tested here a simpler regime-similarity signal currently detects drift at least as usefully. A systematic ablation corrected an earlier hypothesis about which homology dimension is best, and a runtime benchmark shows the real, substantial computational cost of the approach — a cost that the real-data benchmark result makes harder to justify until topology demonstrates a clearer real-data advantage. We present TAMC as an early-stage research prototype, not a production-ready or universally superior method: the evidence base is controlled and synthetic-dominated, the real-data perturbations are injected rather than naturally occurring, the adaptive forecasters used are simple heuristics, and direct comparison against learned non-topological TTA baselines (DynaTTA, COSA, PETSA) remains future work.

## References / Links

This draft does not introduce new external citations beyond what is already catalogued in the repository. For the full annotated bibliography (with URLs), see:

- `paper_notes/related_work.md` — working map of neighboring work by category.
- `paper_notes/research_brief.md`, Section 5 ("Related Work Map") and Section 18 ("Bibliography / Link Index") — full annotations and link index for every paper referenced in Section 8 above.
- `paper_notes/research_brief.md`, Section 19 ("Current Empirical Status") — running log of all numeric results cited in this draft, kept up to date as new experiments are run.
- `paper_notes/introduction.md`, `paper_notes/problem_statement.md`, `paper_notes/methodology.md` — fuller prose versions of Sections 1-3 above.

Repository result artifacts cited in this draft:

- `figures/synthetic_regime_shift_multiseed_metrics.csv`, `figures/logistic_map_shift_multiseed_metrics.csv`, `figures/lorenz_shift_multiseed_metrics.csv` — Section 5.1.
- `figures/tamc_lite_synthetic_forecast_multiseed_metrics.csv` — Section 5.2.
- `figures/real_data_controlled_shift_seasonality_break_multiseed_metrics.csv`, `figures/real_data_controlled_shift_noise_multiseed_metrics.csv` — Section 5.3.
- `figures/topology_ablation_summary.csv`, `figures/topology_ablation_heatmap.png` — Section 6.1.
- `figures/runtime_benchmark_summary.csv`, `figures/runtime_benchmark.png` — Section 7.
