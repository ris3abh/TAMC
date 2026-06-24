# Introduction

## The Problem: Non-Stationary Time-Series Forecasting

Time-series forecasters are typically trained (or otherwise fit) on a
window of historical data and then deployed to predict a stream that keeps
arriving after that window ends. In practice, the dynamics generating that
stream rarely stay fixed forever: weather patterns shift, traffic routing
changes, financial regimes turn over, sensors degrade, physiological state
changes. A forecaster that was accurate on the regime it was fit to can
degrade sharply once the underlying dynamics change — even though nothing
about the forecaster itself changed.

The question this project asks is:

> How can a frozen or black-box time-series forecaster adapt online to
> changing dynamics without labels, gradients, or model-weight updates?

This is the forward-only, label-free test-time adaptation (TTA) setting:
at each time `t`, only the observed history `x_{<=t}` is available — there
is no ground-truth future value to compute a loss against, and the
forecaster's weights are never updated.

## Why Standard Statistics Can Miss the Real Shift

A natural first instinct is to monitor simple statistics of the stream —
rolling mean, rolling variance, autocorrelation, spectral content — and
trigger adaptation when they move. This works well when a shift changes
those statistics directly. But many structurally important shifts do not:

- a periodic signal can become quasi-periodic while keeping the same mean,
  variance, and dominant frequency;
- a system can move from a stable equilibrium into chaos while its
  marginal statistics stay within a similar range;
- a discrete recurrence (e.g. a logistic map) can cross a bifurcation
  threshold into chaotic behavior with first-order statistics that barely
  move.

In each case, what has actually changed is the *shape* of the
dynamics — the structure of the attractor the system's trajectory lives
on — rather than its first- or second-order statistics. A detector that
only watches mean/variance/spectral content is structurally blind to this
class of shift.

## Delay Embeddings Reconstruct Attractor Geometry

Takens' delay embedding theorem gives a way to reconstruct the geometry of
a dynamical system's attractor from a single observed coordinate: stack
delayed copies of the signal,
`[x_t, x_{t+delay}, ..., x_{t+(dimension-1)*delay}]`, into a point cloud
in `R^dimension`. Under mild conditions, this reconstructed point cloud is
diffeomorphic to the system's true (possibly unobserved, possibly
multi-dimensional) phase-space attractor.

This means that even when only one coordinate of a system is observed —
e.g. the `x` component of a 3D Lorenz system — the delay-embedded point
cloud still recovers the qualitative shape of the full attractor: its
loops, its lobes, its clustering structure.

## Persistent Homology as a Structural Drift Signal

Persistent homology summarizes the multi-scale topological structure of a
point cloud: how many connected components it has (H0), how many loops it
has and at what scales they persist (H1), and so on. Applied to a sliding
window of the delay-embedded attractor, it gives a structural fingerprint
of the dynamics in that window — a fingerprint that is sensitive to
exactly the kind of shape change (periodic to quasi-periodic, stable
equilibrium to chaos, recurrence-structure bifurcation) that first-order
statistics can miss.

TAMC compares each window's persistence diagram against a stored
source-regime diagram via Wasserstein distance, producing a scalar drift
score `delta_t` at every step (see
[methodology.md](methodology.md#1-topological-drift-signal) for the exact
pipeline).

## TAMC Uses This Signal to Control Adaptation, Not Just Detect Drift

Detecting drift is necessary but not sufficient. A detector that fires
correctly but is wired into the forecaster naively can still hurt more
than it helps — e.g. an adapter that is always "on" will degrade
in-distribution accuracy even when nothing has shifted. TAMC's contribution
is to use the topological drift signal as a **meta-control** signal: a
gate, rate, or selector that decides *when* and *how much* a forward-only
adapter should act, rather than as a label, a loss term, or a fixed-on
correction (see
[methodology.md](methodology.md#3-tamc-as-meta-control-not-an-additive-loss-term)).

## Summary of Early Findings

Three controlled, synthetic dynamical-systems experiments test whether
topological drift detection generalizes across qualitatively different
kinds of regime shift (10-seed results; see `figures/*_multiseed_metrics.csv`
for full numbers):

- **Sine to quasi-periodic** (clean periodic loop -> added incommensurate
  frequency): TAMC AUROC 0.996 +/- 0.004, delay 11 steps, zero false
  alarms — the cleanest result, and the case TAMC's H1 loop-tracking is
  best suited to.
- **Logistic map** (periodic-ish, `r=3.45` -> chaotic, `r=3.75`): TAMC is
  competitive with the strongest baselines (autocorrelation, spectral) on
  AUROC and delay, and clearly beats naive mean/variance drift detection
  (AUROC ~0.76-0.72 for mean/variance vs >=0.999 for TAMC, autocorrelation,
  and spectral).
- **Lorenz** (stable equilibrium, `rho=20` -> classic chaotic attractor,
  `rho=28`): TAMC beats every baseline on every axis — AUROC 0.964 vs
  0.942 (autocorrelation) and 0.949 (spectral); delay 70.6 steps vs 113
  (autocorrelation) and 117 (spectral); zero false alarms; highest
  separation of all methods.

Beyond detection, one forecasting-adaptation experiment
(`experiments/tamc_lite_synthetic_forecast.py`) tests whether the
topological gate can usefully control a forward-only *adapter* around a
frozen forecaster. Two simple residual-correction adapters
(`MeanShiftResidual`, `AnalogResidualAdapter`) failed to net any benefit —
the shift in that experiment is a frequency/shape change, not a
mean-level shift, so neither residual model had the right inductive bias.
A **topology-gated forecast blend** between the frozen forecaster and a
forward-only adaptive forecaster did work: it achieves the best
**Net Adaptation Score** (0.0257 mean across 10 seeds) of any variant
tested, by preserving most of the frozen forecaster's pre-shift accuracy
while matching an always-on blend's post-shift error reduction. See
[research_brief.md, Section 19](research_brief.md#19-current-empirical-status)
for full numbers and caveats.

These are early, controlled, synthetic results from a research prototype —
not yet a benchmarked method on real data. See
[problem_statement.md](problem_statement.md) for the formal setup and
[research_brief.md](research_brief.md) for the experimental roadmap ahead.
