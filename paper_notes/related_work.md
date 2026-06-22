# Related Work

See [research_biref.md](research_biref.md) for full annotations, links, and the bibliography index. This file tracks the working map of neighboring work and how TAMC differs from each line.

## Forward-only and zeroth-order test-time adaptation

- **FOA** — forward-only adaptation via CMA-ES-optimized prompts, fitness from train/test statistic discrepancy and entropy. TAMC differs by using topological attractor drift as the meta-control signal instead of statistic discrepancy/entropy.
- **FOZO** — forward-only zeroth-order prompt optimization using intermediate feature statistics and entropy. Not time-series-attractor based, no persistent homology.
- **EVA-0** — strict two-forward-pass test-time evolution; raises the efficiency bar. TAMC should report forward-pass budget if compared.

## Time-series test-time adaptation

- **TAFAS** — TSF-TTA with partially observed ground truth and a gated calibration module. TAMC controls adaptation via attractor topology rather than gated calibration alone.
- **PETSA** — parameter-efficient TTA with low-rank adapters, dynamic gating, robust/frequency/structural losses. TAMC can use topology to gate or control similar adapters under dynamical regime shifts.
- **COSA** — lightweight output-space adapter correcting frozen-forecaster predictions from recent context. Closest neighbor to TAMC-Lite's adapter design; TAMC wraps a COSA-style adapter with a topological gate/rate controller.
- **DynaTTA** — real-time shift estimation from prediction error and embedding drift, plus TTFBench. Strongest direct baseline; TAMC substitutes/augments embedding-drift and error signals with persistent-homology attractor drift.
- **Principled TSF-TTA / FAC** — clean protocol based on matured ground truth, avoids leakage. TAMC must define strict unlabeled vs. delayed-feedback variants with the same rigor.
- **Synthetic-to-financial TTA study** — shows aggressive adaptation can hurt in financial markets even when normalization helps on synthetic drift. Motivates using topology to decide *when not* to adapt.

## Topology for time-series forecasting and dynamical systems

- **Topological Attention for Time Series Forecasting** — persistent-homology features as attention signal inside trainable forecasters (e.g., N-BEATS). Blocks broad "topology for forecasting" novelty claims; TAMC uses topology at test time as a control signal, not a training-time feature.
- **SToPS** — persistent homology for delay-embedding parameter (tau) selection. Useful as a method for choosing TAMC's embedding parameters, not a competing contribution.
- **N-BEATS + FastZigzag persistence** — dynamic topological features via zigzag persistence for forecasting. Close to dynamic topology-enhanced forecasting, but not TTA.
- **PH-hybrid deep learning forecasters (PH-RNN/PH-Transformer/PH-CNN)** — topological consistency loss baked into training. TAMC adapts frozen forecasters at test time instead.
- **LISA** — delay-coordinate embeddings + Laplacian spectral learning for inference-time adaptation with frozen decoder and residual adapters. Philosophically closest non-topological neighbor; TAMC differs by using persistent homology rather than spectral/diffusion coordinates.

## Topology for streaming data and drift detection

- **Persistent Homology on Streaming Data** — online/offline PH computation over unbounded streams with bounded summaries. Blocks "first streaming PH" claims.
- **Persistent-entropy landscape-shift detection** — treats concept drift as topological change, uses persistent entropy and topology-preserving projections. Blocks broad "topological concept-drift detection" claims.
- **topoflow** — public project computing PH over sliding-window Takens embeddings with Wasserstein distance for regime-change/periodicity-breakdown detection. Biggest novelty threat to a detector-only framing of TAMC; mitigated by emphasizing the adaptation/control loop, not detection alone.

## Topology-guided TTA outside time-series forecasting

- **Topology-Guided TTA via Persistent Homology (affective behavior)** — PH over neural activation manifolds to predict/avoid harmful adaptation. Blocks "topology-guided TTA is new" claims; TAMC computes topology from reconstructed dynamics, not activations.
- **TopoTTA** — PH-based topological pseudo-labels for anomaly segmentation TTA. Different task domain (segmentation) and topology source (anomaly maps).

## Time-series foundation/backbone models (later-stage candidates)

- **TimesFM**, **Chronos**, **MOMENT**, **Timer** — frozen backbone candidates for prompt/output-space adaptation experiments once the synthetic proof-of-concept (Stage 1-3 in the research brief) is validated.

## Net positioning

No single line of prior work combines: (1) persistent homology, (2) computed from delay-reconstructed attractors, (3) on streaming data, (4) used causally to control, (5) forward-only adaptation, (6) of a frozen time-series forecaster. TAMC's contribution is the closed-loop control mechanism — topological drift gating/rate-controlling/prototype-selecting an adapter — not topological drift detection in isolation.
