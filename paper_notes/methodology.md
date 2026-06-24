# Methodology

This document describes the mechanics of TAMC as currently implemented in
`src/` and exercised by the experiments in `experiments/`. See
[research_brief.md](research_brief.md) for novelty positioning and related
work, and [problem_statement.md](problem_statement.md) for the formal
problem setup.

## 1. Topological Drift Signal

TAMC monitors a streaming univariate signal `x_1, x_2, ..., x_t, ...` by
reconstructing its attractor geometry and comparing that geometry against a
source-regime reference.

Pipeline:

```text
raw signal window
  -> Takens delay embedding (dimension, delay)
  -> point cloud in R^dimension
  -> Vietoris-Rips persistent homology
  -> persistence diagram D_t
  -> Wasserstein distance to source diagram
  -> drift score delta_t
```

Concretely, for a window of length `window` ending at time `t`:

1. Delay-embed the window into a point cloud using `(dimension, delay)`
   (`src/delay_embedding.py::takens_embedding`).
2. Compute the Vietoris-Rips persistence diagram of that point cloud up to
   `max_dimension` (`src/topology_metrics.py::vietoris_rips_persistence`).
3. Extract the diagram for one chosen homology dimension,
   `D_t = diagram_for_dimension(persistence, drift_dimension)`.
4. Score drift as the Wasserstein distance between `D_t` and a stored
   source-regime diagram `D_source`:

```text
delta_t = W_p(D_t, D_source)
```

`D_source` is built once, offline, from a single reference window of the
source/pre-shift regime (`TamicSignal.add_source_prototype`). `delta_t` is
purely a function of the observed window; it never depends on any candidate
adapter parameters (see Section 3 below).

A monitor keeps a running history of `delta_t` values and z-scores new
observations against that history (`TamicSignal.drift_zscore`):

```text
z(delta_t) = (delta_t - mean(history)) / std(history)
```

before the history has accumulated `min_history` points, `z(delta_t)` is
defined to be `0` (no gate action) rather than an unstable early estimate.

## 2. Homology Dimension Choice: H0 vs H1

`drift_dimension` selects which homology dimension's diagram is compared:

- **H0** (`drift_dimension=0`): connected-component birth/death structure.
  Tracks how clustered or spread out the point cloud is, without requiring
  any loop structure.
- **H1** (`drift_dimension=1`): one-dimensional loop birth/death structure.
  Tracks genuine cyclic/rotational structure in the reconstructed attractor.

Across the three Stage 1 detection experiments, the right choice depended
on the *shape* of the source regime's attractor, not on a fixed default:

- **Sine to quasi-periodic** (`synthetic_regime_shift.py`): the source
  regime is a clean periodic loop, so **H1** is the natural and effective
  choice — adding a second incommensurate frequency visibly perturbs the
  loop's persistence structure.
- **Logistic map** (`logistic_map_shift.py`): the source regime is a
  discrete recurrence (period-doubling-cascade dynamics), not a smooth
  loop. **H0** outperformed H1 here — the relevant change is in how the
  point cloud clusters/spreads as the map becomes chaotic, not in loop
  structure.
- **Lorenz** (`lorenz_shift.py`): the source regime settles to a
  noise-dominated near-fixed-point (a compact blob, not a loop), while the
  shifted regime is the classic two-lobe chaotic attractor. **H1
  underperformed** — there is no stable loop to track pre-shift, so its
  drift signal was noisy and at times anti-correlated with the true shift.
  **H0** cleanly separated the compact pre-shift blob from the spread-out,
  multi-lobe post-shift point cloud (10-seed AUROC 0.964, beating both
  autocorrelation and spectral baselines).

**Takeaway:** the homology dimension is itself a modeling choice that
should match the source regime's attractor shape — loopy source regimes
favor H1, compact/non-loopy source regimes favor H0. TAMC does not assume
one dimension works universally; this is reported as an empirical finding,
not a free design parameter to be silently fixed.

## 3. TAMC as Meta-Control, Not an Additive Loss Term

This is the central methodological constraint of the project (see
[research_brief.md, Section 6](research_brief.md#6-methodological-warning-do-not-use-a-constant-topology-term-incorrectly)
for the original warning).

`delta_t` (and any gate or rate derived from it) is computed purely from
the *observed* input stream. It does not depend on which adapter, residual
correction, or forecast candidate is being evaluated at time `t`. This
means:

```text
BAD:  fitness(candidate) = entropy(candidate) + lambda * delta_t
```

is methodologically broken whenever a forward-only/zeroth-order search is
ranking candidates — `delta_t` is a constant across candidates and
contributes nothing to ranking them.

TAMC instead uses `delta_t` only as a **control signal**, multiplying or
selecting among forecasts/corrections rather than appearing as an additive
term in any optimization objective:

```text
g_t       = sigmoid(z(delta_t) - threshold)        # gate in [0, 1]
eta_t     = eta_min + (eta_max - eta_min) * sigmoid(z(delta_t))   # update rate
k_star    = argmin_k distance(D_t, D_source_k)     # prototype selection
```

Every adapter implemented so far (`TamicLiteAdapter`, `TamicBlendPipeline`)
follows this rule: `gate` only ever multiplies a correction or blends two
existing forecasts.

## 4. TAMC-Lite Forecast Blending

The strongest current adaptation result (Section 5 below, and
[research_brief.md, Section 19](research_brief.md#19-current-empirical-status))
comes from **topology-gated forecast blending**
(`src/tamc_pipeline.py::TamicBlendPipeline`), rather than additive residual
correction.

Two forecasters are evaluated on the same context, every step:

- `y_frozen`: a forecaster fit once on the source regime and never updated
  (`LinearARForecaster`).
- `y_adaptive`: a forward-only forecaster that only ever sees the current
  context — no labels, no future data, no gradient updates
  (`RecentPatternForecaster`, which estimates a dominant lag from the
  context's autocorrelation and continues the recent pattern via
  seasonal-naive-with-drift).

The topological gate blends them:

```text
y_blend_t = (1 - g_t) * y_frozen_t + g_t * y_adaptive_t
```

When the gate is near `0` (no detected drift), `y_blend_t ~= y_frozen_t`,
preserving the frozen forecaster's accuracy on the regime it was fit to.
When the gate is near `1` (drift detected), `y_blend_t` shifts toward the
adaptive forecaster, which has no stake in the old regime.

This is in deliberate contrast to two earlier, weaker adapter designs that
were tried first and are kept in the experiment for comparison
(`src/adapters.py::MeanShiftResidual`, `AnalogResidualAdapter`): both add a
*residual correction* to the frozen forecast rather than blending two full
forecasts, and both underperformed the blend (see Section 5).

## 5. Adaptation Tradeoff Metrics

Comparing forecast accuracy pre- and post-shift in isolation does not
capture the central tension every adapter faces: correcting post-shift
error usually risks degrading pre-shift accuracy (since the frozen
forecaster is, by construction, already well-fit to the pre-shift regime).
`experiments/tamc_lite_synthetic_forecast.py` computes four tradeoff
metrics per variant, relative to the frozen forecaster baseline:

```text
pre_harm                  = MAE_pre_variant  - MAE_pre_frozen
post_gain                 = MAE_post_frozen  - MAE_post_variant
net_adaptation_score      = post_gain - pre_harm
post_improvement_pct      = 100 * post_gain / MAE_post_frozen
```

- `pre_harm > 0` means the variant made pre-shift forecasts worse than the
  frozen baseline (the cost of adaptation when nothing has actually
  shifted).
- `post_gain > 0` means the variant made post-shift forecasts better than
  the frozen baseline (the benefit of adaptation).
- `net_adaptation_score` nets the two: it is the single scalar TAMC
  optimizes for in practice, and is the metric that should be compared
  across variants, not `post_improvement_pct` alone.
- For the frozen forecaster itself, all four metrics are exactly `0` by
  construction (it is its own baseline).

`post_improvement_pct` is reported for interpretability but is **noisy
across seeds** — it is a per-seed ratio with a denominator
(`MAE_post_frozen`) that varies substantially seed-to-seed, so its mean and
std can diverge sharply from what the underlying MAE numbers suggest (see
[research_brief.md, Section 19](research_brief.md#19-current-empirical-status)
for the concrete numbers). `net_adaptation_score` does not have this
instability and is the metric to lead with.
