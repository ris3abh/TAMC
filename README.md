# TAMC: Topological Attractor Meta-Control for Forward-Only Time-Series Test-Time Adaptation

TAMC is a research prototype exploring whether persistent homology of delay-reconstructed streaming attractors can serve as a forward-only control signal for adapting time-series forecasting models under non-stationary distribution shift.

The core idea is to monitor the topology of a streaming time series in reconstructed phase space and use topological drift to guide lightweight test-time adaptation without labels, gradients, or model weight updates.

## Research Question

Can the topology of a streaming reconstructed attractor provide a reliable forward-only control signal for adapting forecasting models under non-stationary distribution shift?

## Novelty Positioning

TAMC is not proposed as a generic topological time-series detector or a topology-enhanced forecasting architecture. Prior work has already explored persistent homology for time-series forecasting, streaming topological drift detection, and topology-guided TTA in vision-like settings. TAMC focuses on the intersection that remains underexplored: using persistent homology of streaming delay-reconstructed attractors as a causal meta-control signal for forward-only test-time adaptation of frozen time-series forecasters.

To our knowledge, TAMC is the first framework to explicitly couple streaming persistent homology of delay-reconstructed time-series attractors with forward-only test-time adaptation of frozen forecasting models under non-stationary distribution shift. Unlike prior topology-based forecasting methods, TAMC does not train topology into the forecaster; unlike prior time-series TTA methods, it controls adaptation using dynamical attractor topology; and unlike topology-guided TTA in vision, it computes topology from the reconstructed data-generating dynamics rather than from neural activation manifolds.

See [paper_notes/research_brief.md](paper_notes/research_brief.md) for the full novelty assessment, related-work map, and experimental roadmap.

## Pipeline

Streaming signal → Delay-coordinate embedding → Sliding point cloud → Persistent homology → Topological drift score → Forward-only adapter → Adapted forecast

## Status

Early-stage research prototype.

## Current Results

- Stage 1 detection experiments (sine to quasi-periodic, logistic map,
  Lorenz attractor) show that the TAMC topological drift signal is
  competitive with, or better than, statistical (mean/variance),
  autocorrelation, and spectral baselines on controlled synthetic and
  dynamical-systems regime shifts.
- TAMC-Lite forecast blending currently improves the adaptation tradeoff
  on a sine-to-quasi-periodic forecasting task: a topology-gated blend
  between a frozen forecaster and a forward-only adaptive forecaster
  achieves the best Net Adaptation Score of any variant tested, by
  preserving pre-shift accuracy better than an always-on blend while
  matching its post-shift improvement.
- The same topology-gated blend also has the best adaptation tradeoff on a
  real series (ETTh1, `OT` column) under a controlled, injected
  perturbation (`seasonality_break`) — ahead of three non-topological
  baselines (autocorrelation-, spectral-, and mean/variance-gated)
  compared under the identical gate control law, though the margin there
  is modest, not dramatic.
- A systematic ablation over homology dimension (H0 vs H1), embedding
  delay, and topology window size, across all three detection systems,
  shows H0 as the more robust default everywhere tested — H1 is
  competitive on loop-like attractors (sine, logistic map) but never
  clearly better and has its own failure modes, and is both weaker and
  far less stable on Lorenz's compact-to-chaotic transition. This revises
  an earlier, weaker claim that was based on one hand-picked dimension
  per system rather than a full sweep.
- A runtime benchmark quantifies TAMC's compute cost honestly: at
  window=192, computing TAMC's topological drift takes roughly four
  orders of magnitude longer per window than spectral or autocorrelation
  drift (~13,000x and ~2,300x respectively), and the gap grows steeply
  with window size since persistent homology scales poorly with
  point-cloud size. At the smaller windows (<=128) used throughout this
  repo's detection experiments, the absolute cost is still small (tens of
  milliseconds per window), but this would not scale to large windows or
  per-step scoring without further optimization.
- This is still an early research prototype evaluated only on controlled
  synthetic dynamical systems and controlled, injected perturbations on
  one real series — not yet tested on a naturally occurring real-world
  regime shift, and not a final benchmarked method. See
  [paper_notes/research_brief.md, Section 19](paper_notes/research_brief.md#19-current-empirical-status)
  for full numbers and current limitations.

## Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for Python version management, dependency locking, virtual environments, and command execution.

### Setup

```bash
uv python install 3.11
uv sync
```

# Usage

```bash
make sync
make test
make lint
make format
make notebook
```

## Repository structure

- `paper_notes/` — drafts and notes for the accompanying paper
- `src/` — core implementation
- `experiments/` — reproducible experiment scripts and notebooks
- `data/` — datasets (not tracked in git)
- `figures/` — generated figures (not tracked in git)

## License

MIT — see [LICENSE](LICENSE).
