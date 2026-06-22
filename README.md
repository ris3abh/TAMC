# TAMC

# TAMC: Topological Attractor Meta-Control for Forward-Only Time-Series Test-Time Adaptation

TAMC is a research prototype exploring whether persistent homology of delay-reconstructed streaming attractors can serve as a forward-only control signal for adapting time-series forecasting models under non-stationary distribution shift.

The core idea is to monitor the topology of a streaming time series in reconstructed phase space and use topological drift to guide lightweight test-time adaptation without labels, gradients, or model weight updates.

## Research Question

Can the topology of a streaming reconstructed attractor provide a reliable forward-only control signal for adapting forecasting models under non-stationary distribution shift?

## Pipeline

Streaming signal → Delay-coordinate embedding → Sliding point cloud → Persistent homology → Topological drift score → Forward-only adapter → Adapted forecast

## Status

Early-stage research prototype.

## Repository structure

- `paper_notes/` — drafts and notes for the accompanying paper
- `src/` — core implementation
- `experiments/` — notebooks for experiments and demos
- `data/` — datasets (not tracked in git)
- `figures/` — generated figures (not tracked in git)

## License

MIT — see [LICENSE](LICENSE).
