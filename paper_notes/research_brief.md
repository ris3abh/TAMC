# TAMC Research Brief

**Project:** Topological Attractor Meta-Control (TAMC) for Forward-Only Time-Series Test-Time Adaptation  
**Repo status:** Early-stage research prototype  
**Purpose of this document:** Summarize the current novelty assessment, related work, risks, implementation plan, and publication roadmap for the TAMC project.

---

## 1. Executive Summary

TAMC is a research direction exploring whether persistent homology of streaming delay-reconstructed attractors can serve as a causal, forward-only control signal for adapting frozen time-series forecasting models under non-stationary distribution shift.

The key idea is:

```text
Streaming signal
  -> delay-coordinate embedding
  -> sliding attractor point cloud
  -> persistent homology
  -> topological drift score
  -> forward-only meta-control
  -> adapted forecast
```

The current novelty assessment is:

> The detector component alone is not fully novel. Topology for time-series forecasting is not fully novel. Topology-guided test-time adaptation is not fully novel. However, the specific closed-loop formulation - persistent homology of streaming delay-reconstructed attractors used as a causal, forward-only meta-control signal for prompt, activation, or output-space adaptation of frozen time-series forecasters - appears to be a defensible and publishable research gap.

The recommended novelty claim is:

> To our knowledge, TAMC is the first framework to explicitly couple streaming persistent homology of delay-reconstructed time-series attractors with forward-only test-time adaptation of frozen forecasting models under non-stationary distribution shift.

A safer version for publication:

> TAMC studies whether persistent homology of streaming delay-reconstructed attractors can serve as a causal meta-control signal for forward-only test-time adaptation of frozen time-series forecasters. Unlike prior topology-based forecasting methods, TAMC does not train topology into the forecaster; unlike prior time-series TTA methods, it controls adaptation using dynamical attractor topology; and unlike topology-guided TTA in vision, it computes topology from the reconstructed data-generating dynamics rather than from neural activation manifolds.

---

## 2. What Problem Are We Solving?

Modern time-series forecasters are often trained on historical data but deployed into environments that change over time. This creates non-stationary test-time conditions. A model that performs well on the training distribution may degrade when the underlying dynamics of the stream shift.

Examples include:

- energy demand under changing weather and behavior patterns,
- traffic under new routing or events,
- financial markets under regime changes,
- industrial sensors under equipment degradation,
- physiological signals under changing patient state,
- cyber-physical systems with changing control dynamics.

The central problem is:

> How can a frozen or black-box time-series forecaster adapt online to changing dynamics without labels, gradients, or model-weight updates?

Current test-time adaptation methods often rely on prediction entropy, confidence, recent error, normalization, feature-statistic mismatch, or embedding drift. These signals are useful, but they do not directly measure whether the underlying dynamical system has changed shape.

TAMC proposes to monitor the reconstructed attractor topology of the streaming signal and use changes in that topology to control adaptation.

---

## 3. Why Topology?

A time series can maintain similar mean, variance, or marginal distribution while changing its dynamical structure. For example:

- periodic behavior can become quasi-periodic,
- stable oscillation can become chaotic,
- one seasonal recurrence structure can become another,
- a system can preserve amplitude but change recurrence geometry,
- a stream can keep similar first-order statistics while changing autocorrelation, loops, or attractor geometry.

Persistent homology can summarize multi-scale topological structure in the delay-reconstructed phase space. In practical terms, it can detect changes in connected components, loops, holes, and persistence lifetimes.

The central hypothesis is:

> Topological drift of the reconstructed streaming attractor provides a structurally meaningful signal for when and how a frozen time-series forecaster should adapt at test time.

---

## 4. Novelty Boundaries

### 4.1 Claims We Should Not Make

Do **not** claim:

1. "We are the first to use persistent homology for time-series forecasting."  
   This is false. Topological Attention for Time Series Forecasting already used persistent homology as a forecasting signal.

2. "We are the first to use delay embeddings and persistent homology on time series."  
   This is false. Delay embeddings plus persistent homology are established in topological time-series analysis.

3. "We are the first to use topology for test-time adaptation."  
   This is false. Recent work has applied persistent homology to topology-guided TTA in vision/activation manifolds and anomaly segmentation.

4. "We are the first to compute persistent homology on streaming data."  
   This is false. There is prior work on persistent homology for streaming data.

5. "We are the first to detect topological drift in streams."  
   This is too broad. There are prior topological drift and persistent-entropy approaches, and a public project called `topoflow` already computes persistent homology over sliding-window Takens embeddings for streaming time-series regime-change detection.

### 4.2 Claims We Can Make

The strongest defensible claim is:

> TAMC is a forward-only time-series test-time adaptation framework that uses persistent homology of streaming delay-reconstructed attractors as a meta-control signal for adapting frozen forecasting models.

The contribution is the closed-loop control mechanism:

```text
Observed stream topology -> topological drift -> adaptation gate/rate/prototype/search control -> forecast correction
```

The novelty is not just detecting drift. The novelty is using topological attractor drift to control adaptation.

---

## 5. Related Work Map

### 5.1 Forward-Only Test-Time Adaptation

#### FOA: Test-Time Model Adaptation with Only Forward Passes

- Link: https://arxiv.org/abs/2404.01650
- HTML: https://arxiv.org/html/2404.01650v1
- Summary: FOA adapts models using only forward passes. It learns prompts through derivative-free CMA-ES and uses a fitness function based on training-test statistic discrepancy and prediction entropy.
- Why it matters: This is the most important forward-only adaptation baseline.
- TAMC difference: FOA uses statistic discrepancy and entropy; TAMC uses topological attractor drift as a meta-control signal.

#### FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation

- Link: https://arxiv.org/abs/2603.04733
- CVPR 2026 PDF: https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_FOZO_Forward-Only_Zeroth-Order_Prompt_Optimization_for_Test-Time_Adaptation_CVPR_2026_paper.pdf
- Code: https://github.com/eVI-group-SCU/FOZO
- Summary: FOZO proposes forward-only zeroth-order prompt optimization for TTA. It optimizes prompts using intermediate feature statistics and prediction entropy.
- Why it matters: This is a strong modern forward-only baseline.
- TAMC difference: FOZO is not time-series-attractor based and does not use persistent homology.

#### EVA-0: Test-Time Model Evolution with Only Two Forward Passes per Sample

- Link: https://arxiv.org/abs/2605.18867
- Summary: EVA-0 studies strict two-forward-pass test-time evolution and addresses shortcut solutions, weight drift, and noisy update-direction estimation.
- Why it matters: It raises the efficiency bar for forward-only/zeroth-order adaptation.
- TAMC difference: TAMC should report runtime and forward-pass budget carefully if compared to newer forward-only methods.

---

### 5.2 Time-Series Test-Time Adaptation

#### TAFAS: Battling the Non-stationarity in Time Series Forecasting via Test-Time Adaptation

- Link: https://arxiv.org/abs/2501.04970
- Code: https://github.com/kimanki/TAFAS
- Summary: TAFAS introduces a TSF-TTA framework using partially observed ground truth and a gated calibration module.
- Why it matters: Direct time-series TTA baseline.
- TAMC difference: TAMC controls adaptation using reconstructed attractor topology rather than only gated calibration and partially observed targets.

#### PETSA: Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting

- Link: https://arxiv.org/abs/2506.23424
- OpenReview: https://openreview.net/forum?id=uFj4EL4GTB
- Code: https://github.com/BorealisAI/PETSA
- Summary: PETSA adapts time-series forecasters using small input/output calibration modules, low-rank adapters, dynamic gating, and losses with robust, frequency, and structural components.
- Why it matters: Strong parameter-efficient TTA baseline.
- TAMC difference: TAMC can use topology to gate or control similar adapters, especially under dynamical regime shifts.

#### COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting

- OpenReview: https://openreview.net/forum?id=L7Z5wBMPrW
- PDF: https://openreview.net/pdf?id=L7Z5wBMPrW
- Code: https://github.com/bigbases/COSA_ICLR2026
- Summary: COSA uses a lightweight output-space adapter to directly correct predictions of frozen forecasters using recent context.
- Why it matters: Very close to TAMC-Lite's likely output-adapter design.
- TAMC difference: TAMC can wrap a COSA-style output adapter with a topological gate or topology-derived update-rate controller.

#### DynaTTA: Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting

- OpenReview: https://openreview.net/forum?id=a399SmgWGl
- PDF: https://openreview.net/pdf?id=a399SmgWGl
- Code: https://github.com/shivam-grover/DynaTTA
- Summary: DynaTTA estimates shifts in real time using prediction errors and embedding drift, then controls adaptation. It also introduces TTFBench for controlled time-series forecasting shifts.
- Why it matters: One of the strongest direct baselines for TAMC.
- TAMC difference: TAMC uses persistent-homology attractor drift instead of, or in addition to, prediction-error and embedding-drift signals.

#### Towards Principled Test-Time Adaptation for Time Series Forecasting

- Link: https://arxiv.org/abs/2605.17250
- Summary: This paper proposes a cleaner TSF-TTA protocol based on matured ground truth and introduces Frequency-Aware Calibration.
- Why it matters: Important for avoiding leakage and establishing a clean adaptation protocol.
- TAMC difference: TAMC should define strict unlabeled and delayed-feedback variants clearly.

#### Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets

- Link: https://arxiv.org/abs/2602.00073
- Summary: Studies small-footprint TTA for causal time-series forecasting and direction classification; finds that simple normalization-based adaptation can help in synthetic drift but more aggressive adaptation can hurt in financial markets.
- Why it matters: Useful practical caution against over-adaptation.
- TAMC difference: TAMC can use topology to decide when not to adapt.

---

### 5.3 Topology for Time-Series Forecasting and Dynamical Systems

#### Topological Attention for Time Series Forecasting

- NeurIPS page: https://proceedings.neurips.cc/paper/2021/hash/d062f3e278a1fbba2303ff5a22e8c75e-Abstract.html
- ArXiv: https://arxiv.org/abs/2107.09031
- OpenReview: https://openreview.net/forum?id=Xl1Z1L9DBIJ
- Summary: Uses local persistent-homology features as attention signals for forecasting, integrating them with trainable forecasting models such as N-BEATS.
- Why it matters: Blocks broad novelty claims about topology for forecasting.
- TAMC difference: TAMC uses topology at test time as a control signal, not as a training-time attention feature.

#### SToPS: Persistent Homology for Time-Delay Embedding Parameter Selection

- Link: https://arxiv.org/abs/2302.03447
- Summary: Uses persistent homology to identify dynamically significant time lags for delay embeddings.
- Why it matters: Useful for choosing the delay parameter `tau` and embedding dimension in TAMC.
- TAMC difference: TAMC uses delay embeddings for streaming adaptation control, not just embedding-parameter selection.

#### N-BEATS + FastZigzag Persistence

- Link: https://link.springer.com/chapter/10.1007/978-3-032-21628-1_15
- Summary: Uses time-delay embeddings and FastZigzag persistence to derive dynamic topological features for N-BEATS forecasting.
- Why it matters: Very close to dynamic topology-enhanced forecasting.
- TAMC difference: TAMC is not topology-feature forecasting; it is forward-only TTA meta-control.

#### Persistent-Homology Hybrid Deep Learning Forecasting

- Link: https://link.springer.com/article/10.1007/s40747-026-02315-2
- Summary: Uses persistent-homology feature extraction and topological consistency loss in PH-RNN, PH-Transformer, and PH-CNN forecasting models.
- Why it matters: Shows topology for forecasting is active and competitive.
- TAMC difference: TAMC adapts frozen forecasters at test time rather than training a topology-aware model.

#### LISA: Laplacian In-context Spectral Analysis

- Link: https://arxiv.org/abs/2602.04906
- Summary: Uses delay-coordinate embeddings and Laplacian spectral learning for inference-time adaptation with a frozen nonlinear decoder and lightweight latent-space residual adapters.
- Why it matters: Philosophically close because it uses delay embeddings and changing dynamics for inference-time adaptation.
- TAMC difference: LISA is spectral/diffusion-coordinate based; TAMC is persistent-homology-based meta-control.

---

### 5.4 Topology for Streaming Data and Drift Detection

#### Persistent Homology on Streaming Data

- NSF page: https://par.nsf.gov/servlets/purl/10350969
- IEEE page: https://ieeexplore.ieee.org/document/9346556
- Summary: Proposes online/offline persistent-homology computation for potentially unbounded evolving streams using a bounded summary.
- Why it matters: Blocks claims that TAMC is the first streaming PH method.
- TAMC difference: TAMC applies topology to forecasting adaptation control.

#### Unsupervised Assessment of Landscape Shifts Based on Persistent Entropy and Topological Preservation

- ArXiv: https://arxiv.org/abs/2410.04183
- Springer: https://link.springer.com/chapter/10.1007/978-3-031-82346-6_9
- Summary: Treats concept drift as including changes in topological characteristics and uses persistent entropy plus topology-preserving projections.
- Why it matters: Blocks broad claims about topological concept-drift detection.
- TAMC difference: TAMC uses topological drift as a control signal for forecasting adaptation.

#### topoflow

- Repo: https://github.com/techmede/topoflow
- README: https://github.com/techmede/topoflow/blob/main/README.md
- Summary: Public project computing persistent homology over sliding-window Takens embeddings and Wasserstein distance between diagrams to detect regime change, periodicity breakdown, and structural anomalies.
- Why it matters: Biggest novelty threat to detector-only TAMC.
- TAMC difference: TAMC must emphasize adaptation/control, not only detection.

---

### 5.5 Topology-Guided Test-Time Adaptation Outside Time-Series Forecasting

#### Topology-Guided Test-Time Adaptation via Persistent Homology

- PDF: https://openaccess.thecvf.com/content/CVPR2026W/ABAW/papers/Mutlu_Topology-Guided_Test-Time_Adaptation_via_Persistent_Homology_From_Affective_Behavior_Analysis_CVPRW_2026_paper.pdf
- Summary: Uses persistent homology over neural activation manifolds to predict adaptation outcomes and avoid harmful adaptation.
- Why it matters: Blocks claims that topology-guided TTA is new.
- TAMC difference: TAMC computes topology from delay-reconstructed time-series attractors, not from neural activation manifolds.

#### TopoTTA: Topology-Aware Test-Time Adaptation for Anomaly Segmentation

- Project: https://topotta.github.io/
- Code: https://github.com/EngrUsmaanAli/TopoTTA
- Summary: Uses persistent homology in TTA for anomaly segmentation, deriving topological pseudo-labels from anomaly maps.
- Why it matters: Another topology + TTA neighbor.
- TAMC difference: TAMC targets non-stationary time-series forecasting and uses attractor topology.

---

### 5.6 Time-Series Foundation/Backbone Models

These should be considered later, after the prototype works with simple models.

#### TimesFM

- Link: https://arxiv.org/abs/2310.10688
- Summary: Decoder-only time-series forecasting foundation model.
- TAMC usage: Potential frozen backbone after proof-of-concept.

#### Chronos

- Link: https://arxiv.org/abs/2403.07815
- Summary: Tokenizes time-series values and trains language-model architectures for forecasting.
- TAMC usage: Good candidate for prompt/output adaptation experiments.

#### MOMENT

- Link: https://arxiv.org/abs/2402.03885
- Summary: Open family of time-series foundation models.
- TAMC usage: Frozen backbone candidate.

#### Timer

- Link: https://arxiv.org/abs/2402.02368
- Summary: Generative pretrained transformer for time-series analysis.
- TAMC usage: Frozen backbone candidate.

---

## 6. Methodological Warning: Do Not Use a Constant Topology Term Incorrectly

A major issue must be handled carefully.

If the topological distance is computed only from the observed input stream:

```text
delta_t = distance(D_t, D_source)
```

then `delta_t` is constant across candidate adapter parameters during zeroth-order optimization. Adding this directly to a CMA-ES or SPSA fitness function will not help rank candidate prompts or adapters.

Bad objective:

```text
fitness(candidate) = entropy(candidate) + lambda * delta_t
```

If `delta_t` does not depend on `candidate`, it is just a constant.

Better uses:

1. Use topology as a gate.
2. Use topology as an adaptation-rate controller.
3. Use topology to select a source/regime prototype.
4. Make topology candidate-dependent by computing topology on predicted continuations generated by each candidate adapter.

---

## 7. Recommended TAMC Variants

### 7.1 TAMC-Lite: Topology-Gated Output Correction

This should be the first implementation.

Frozen forecaster:

```text
y_hat_t = f_theta(x_{t-L:t})
```

Adapted forecast:

```text
y_hat_TAMC_t = y_hat_t + g_t * r_phi(context_t)
```

where:

```text
g_t = topology-derived gate
r_phi = lightweight residual correction
```

This is easy to compare against COSA-style adapters and DynaTTA-style shift gates.

### 7.2 TAMC-Rate: Topology-Controlled Update Rate

Use topological drift to control how aggressively an adapter updates:

```text
eta_t = eta_min + (eta_max - eta_min) * sigmoid((delta_t - mu_delta) / sigma_delta)
```

This makes TAMC a true meta-controller.

### 7.3 TAMC-Prototype: Topological Source Prototype Matching

Store multiple source/regime topology prototypes:

```text
D_source_1, D_source_2, ..., D_source_K
```

At test time:

```text
k_star = argmin_k distance(D_t, D_source_k)
```

Then select an adapter, calibration rule, or bias associated with the closest topological regime.

### 7.4 TAMC-FO: Candidate-Dependent Forward-Only Adaptation

For a candidate adapter `a`, generate a forecast continuation:

```text
x_t history + y_hat_candidate_future
```

Then compute topology over the combined observed-plus-predicted segment and penalize candidate forecasts that create implausible topology.

This is more expensive but makes the topological objective candidate-dependent.

---

## 8. Experimental Plan

### Stage 1: Topological Drift Detection

Goal:

> Show that topological attractor drift detects structural regime changes that ordinary statistics miss.

Datasets/systems:

1. Sine to quasi-periodic sine
2. Logistic map parameter shift
3. Lorenz system parameter shift
4. Mackey-Glass chaotic series
5. AR process with matched marginal variance

Metrics:

- detection delay,
- false alarm rate,
- AUROC for regime shift windows,
- F1 score for boundary detection,
- correlation with future forecast-error increase.

Baselines:

- rolling mean,
- rolling variance,
- autocorrelation distance,
- FFT/periodogram distance,
- MMD,
- CUSUM/Page-Hinkley,
- embedding drift,
- prediction error when delayed labels are allowed.

Deliverable:

> Figure 1: topological drift rises near structural regime changes while mean/variance remain stable.

### Stage 2: Frozen Forecaster Baseline

Train or use a frozen forecaster on the source regime only.

Suggested first models:

- LSTM,
- TCN,
- N-BEATS,
- PatchTST,
- DLinear.

Goal:

> Show that frozen forecasting error increases after a structural regime shift.

### Stage 3: TAMC-Lite Adapter

Implement topology-gated residual output correction.

Compare:

- frozen model,
- output adapter without topology,
- mean/variance-gated adapter,
- spectral-gated adapter,
- embedding-drift-gated adapter,
- TAMC-gated adapter.

Goal:

> Show TAMC reduces post-shift forecast error or prevents harmful adaptation.

### Stage 4: Forward-Only Adaptation

Implement SPSA or CMA-ES over small adapter parameters:

- output bias,
- residual scale,
- normalization affine parameters,
- prompt vector,
- lightweight linear adapter.

Use topology to control:

- update rate,
- update frequency,
- gate strength,
- search radius,
- reset/freeze decision.

### Stage 5: Real Benchmarks

Datasets:

- ETT,
- Electricity,
- Weather,
- Traffic,
- Exchange,
- selected Monash Forecasting Archive datasets,
- financial/crypto series only after synthetic proof.

Use DynaTTA/TTFBench-style controlled perturbations:

- trend shifts,
- seasonality shifts,
- regime shifts,
- localized noise,
- periodicity breakdown.

---

## 9. Baselines Reviewers Will Expect

Minimum baselines:

1. Frozen forecaster
2. Oracle regime-aware model or retrained upper bound
3. Rolling mean/variance drift gate
4. Autocorrelation drift gate
5. Spectral/FFT drift gate
6. MMD drift gate
7. RevIN-style normalization
8. TAFAS-style gated calibration
9. PETSA-style lightweight adapter
10. COSA-style output adapter
11. DynaTTA-style embedding-drift gate
12. FOA/FOZO-style forward-only adaptation
13. Topology-only detector without adaptation
14. Random gate control
15. Oracle shift-boundary gate

The most important comparisons are:

- TAMC vs DynaTTA-style embedding drift
- TAMC vs COSA-style output adapter
- TAMC vs PETSA-style adapter
- TAMC vs spectral/autocorrelation drift
- TAMC vs FOA/FOZO-inspired forward-only adaptation

---

## 10. Risks and Mitigations

### Risk 1: Topology is unnecessary

Reviewers may ask why autocorrelation, FFT, MMD, or embedding drift are not enough.

Mitigation:

- Include strong non-topological baselines.
- Show cases where topology helps most: periodic-to-quasi-periodic, periodic-to-chaotic, attractor structure changes, recurrence breakdown.

### Risk 2: Topology detects shift but does not improve forecasting

Detection alone is not enough.

Mitigation:

- Show adaptation improvement, not just drift detection.
- Use topology as a gate to prevent harmful adaptation.

### Risk 3: Topological distance lacks direction

A scalar topological distance says how different the current regime is, not what correction to apply.

Mitigation:

- Use topology for meta-control: gate, update rate, prototype selection, reset/freeze, or search radius.
- Do not use topology as a direct signed correction unless learned from data.

### Risk 4: Persistent homology is expensive

Mitigation:

- Start with H0 and H1 only.
- Use small sliding windows: 32, 64, 128.
- Compute topology every k steps.
- Use Ripser or GUDHI with sparse/edge-collapse options.
- Use persistence images, landscapes, Betti curves, or entropy for fast summaries.

### Risk 5: Embedding parameters are sensitive

Mitigation:

- Ablate tau, embedding dimension, and window size.
- Compare mutual information, false nearest neighbors, and SToPS-like selection.
- Use train-only normalization.

### Risk 6: Real-world streams may not have clean topology

Mitigation:

- Start with synthetic dynamical systems.
- Report where topology helps and where it does not.
- Position TAMC as strongest under structural/dynamical shifts, not all possible shifts.

### Risk 7: Leakage in time-series TTA

Mitigation:

Define two settings clearly:

1. Strict unlabeled TAMC: uses only x_{<=t}.
2. Delayed-feedback TAMC: uses ground truth only after it naturally becomes observable.

Never mix these protocols.

### Risk 8: Novelty threat from topoflow

Mitigation:

Position TAMC as an adaptation framework, not a detector-only library.

---

## 11. Repo Roadmap

The repo currently has the right scaffold:

```text
paper_notes/
src/
experiments/
data/
figures/
requirements.txt
README.md
```

Next files to implement:

### `src/delay_embedding.py`

Implement:

- univariate Takens embedding,
- multivariate delay embedding,
- sliding window extraction,
- train-only normalization,
- validation for tau, dimension, and window length.

### `src/topology_metrics.py`

Implement:

- Vietoris-Rips persistent homology,
- H0 and H1 extraction,
- persistence entropy,
- total persistence,
- max persistence,
- Betti curves,
- bottleneck distance,
- Wasserstein distance,
- persistence images or landscapes.

Useful libraries:

- GUDHI: https://gudhi.inria.fr/python/latest/rips_complex_user.html
- Ripser.py: https://ripser.scikit-tda.org/
- Persim: https://github.com/scikit-tda/persim
- giotto-tda: https://giotto-ai.github.io/gtda-docs/latest/notebooks/topology_time_series.html

### `src/tamic_signal.py`

Implement:

- source topology prototype builder,
- sliding online topology computation,
- topological drift score,
- MAD/z-score thresholding,
- topology gate,
- source-prototype matching.

### `src/adapters.py`

Implement:

- frozen model wrapper,
- output residual adapter,
- topology-gated residual,
- zeroth-order adapter interface,
- CMA-ES/SPSA-compatible objective.

### `experiments/synthetic_regime_shift.ipynb`

First experiment:

- generate synthetic regime shift,
- compute topological drift,
- compare with mean/variance/autocorrelation/spectral drift,
- plot all signals.

### `experiments/sine_quasiperiodic_shift.ipynb`

Best first demo.

### `experiments/lorenz_shift.ipynb`

Best dynamical-systems credibility demo.

---

## 12. Suggested `requirements.txt` Additions

Current requirements are a good start. Add:

```text
pandas
torch
tqdm
ripser
persim
statsmodels
ruptures
pytest
black
ruff
```

Optional later:

```text
giotto-tda
cma
jax
optuna
```

---

## 13. Paper Structure

Suggested title:

> TAMC: Topological Attractor Meta-Control for Forward-Only Test-Time Adaptation in Non-Stationary Time-Series Forecasting

Suggested sections:

1. Introduction
2. Related Work
3. Problem Setup
4. Topological Attractor Drift
5. TAMC Meta-Control
6. Forward-Only Adapter Variants
7. Experiments
8. Ablations
9. Limitations
10. Conclusion

Main figures:

1. TAMC pipeline diagram
2. Topological drift vs mean/variance/spectral drift
3. Forecasting performance before and after shift
4. Adapter gate over time
5. Persistence diagrams before and after shift
6. Runtime/accuracy tradeoff
7. Ablation heatmap over tau, dimension, and window size

---

## 14. Publication Strategy

### First target: workshop paper

Good first submission angle:

> TAMC-Lite: topology-gated output adaptation under controlled dynamical regime shifts.

Potential workshop areas:

- time-series representation learning,
- test-time adaptation,
- distribution shift,
- AI for dynamical systems,
- topology and geometry in machine learning,
- robust forecasting.

### Full paper target

A full paper needs:

- strong direct baselines,
- real datasets,
- controlled shift benchmark,
- no-leakage protocol,
- runtime analysis,
- ablations,
- open-source code.

Potential venues:

- NeurIPS,
- ICML,
- ICLR,
- KDD,
- AAAI,
- IJCAI,
- TMLR,
- applied ML/time-series venues.

---

## 15. Immediate Next Steps

1. Update README with sharper novelty claim.
2. Add this document as `paper_notes/research_brief.md`.
3. Expand `paper_notes/related_work.md` using Section 5 above.
4. Implement `src/delay_embedding.py`.
5. Implement `src/topology_metrics.py`.
6. Build the first synthetic drift notebook.
7. Produce Figure 1: topology vs statistics under matched mean/variance shift.
8. Add TAMC-Lite output adapter.
9. Compare against non-topological gates.
10. Only then move to larger forecasters or foundation models.

---

## 16. One-Sentence Project Definition

> TAMC is a forward-only test-time adaptation framework that monitors the topology of a streaming delay-reconstructed attractor and uses topological drift as a meta-control signal for adapting frozen time-series forecasting models under non-stationary dynamics.

---

## 17. Short README Blurb

You can add this to the repo README:

```markdown
## Novelty Positioning

TAMC is not proposed as a generic topological time-series detector or a topology-enhanced forecasting architecture. Prior work has already explored persistent homology for time-series forecasting, streaming topological drift detection, and topology-guided TTA in vision-like settings. TAMC focuses on the intersection that remains underexplored: using persistent homology of streaming delay-reconstructed attractors as a causal meta-control signal for forward-only test-time adaptation of frozen time-series forecasters.
```

---

## 18. Bibliography / Link Index

### Forward-only and zeroth-order TTA

- FOA: https://arxiv.org/abs/2404.01650
- FOA HTML: https://arxiv.org/html/2404.01650v1
- FOZO: https://arxiv.org/abs/2603.04733
- FOZO CVPR PDF: https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_FOZO_Forward-Only_Zeroth-Order_Prompt_Optimization_for_Test-Time_Adaptation_CVPR_2026_paper.pdf
- FOZO code: https://github.com/eVI-group-SCU/FOZO
- EVA-0: https://arxiv.org/abs/2605.18867

### Time-series TTA

- TAFAS: https://arxiv.org/abs/2501.04970
- TAFAS code: https://github.com/kimanki/TAFAS
- PETSA: https://arxiv.org/abs/2506.23424
- PETSA OpenReview: https://openreview.net/forum?id=uFj4EL4GTB
- PETSA code: https://github.com/BorealisAI/PETSA
- COSA OpenReview: https://openreview.net/forum?id=L7Z5wBMPrW
- COSA PDF: https://openreview.net/pdf?id=L7Z5wBMPrW
- COSA code: https://github.com/bigbases/COSA_ICLR2026
- DynaTTA OpenReview: https://openreview.net/forum?id=a399SmgWGl
- DynaTTA PDF: https://openreview.net/pdf?id=a399SmgWGl
- DynaTTA code: https://github.com/shivam-grover/DynaTTA
- Principled TSF-TTA / FAC: https://arxiv.org/abs/2605.17250
- Non-stationary TTA synthetic-to-finance: https://arxiv.org/abs/2602.00073

### Topology for time-series and forecasting

- Topological Attention for Time Series Forecasting: https://proceedings.neurips.cc/paper/2021/hash/d062f3e278a1fbba2303ff5a22e8c75e-Abstract.html
- Topological Attention arXiv: https://arxiv.org/abs/2107.09031
- Topological Attention OpenReview: https://openreview.net/forum?id=Xl1Z1L9DBIJ
- SToPS / PH for delay embeddings: https://arxiv.org/abs/2302.03447
- N-BEATS + FastZigzag persistence: https://link.springer.com/chapter/10.1007/978-3-032-21628-1_15
- PH hybrid deep learning forecasting: https://link.springer.com/article/10.1007/s40747-026-02315-2
- LISA: https://arxiv.org/abs/2602.04906

### Streaming topology and drift

- Persistent Homology on Streaming Data: https://par.nsf.gov/servlets/purl/10350969
- Persistent Homology on Streaming Data, IEEE: https://ieeexplore.ieee.org/document/9346556
- Persistent entropy landscape shift: https://arxiv.org/abs/2410.04183
- Persistent entropy Springer chapter: https://link.springer.com/chapter/10.1007/978-3-031-82346-6_9
- topoflow repo: https://github.com/techmede/topoflow
- topoflow README: https://github.com/techmede/topoflow/blob/main/README.md

### Topology-guided TTA outside forecasting

- Topology-Guided TTA via Persistent Homology: https://openaccess.thecvf.com/content/CVPR2026W/ABAW/papers/Mutlu_Topology-Guided_Test-Time_Adaptation_via_Persistent_Homology_From_Affective_Behavior_Analysis_CVPRW_2026_paper.pdf
- TopoTTA project: https://topotta.github.io/
- TopoTTA code: https://github.com/EngrUsmaanAli/TopoTTA

### Time-series foundation models

- TimesFM: https://arxiv.org/abs/2310.10688
- Chronos: https://arxiv.org/abs/2403.07815
- MOMENT: https://arxiv.org/abs/2402.03885
- Timer: https://arxiv.org/abs/2402.02368

### TDA libraries

- GUDHI Rips complex docs: https://gudhi.inria.fr/python/latest/rips_complex_user.html
- Ripser.py: https://ripser.scikit-tda.org/
- Persim: https://github.com/scikit-tda/persim
- giotto-tda time-series topology tutorial: https://giotto-ai.github.io/gtda-docs/latest/notebooks/topology_time_series.html

---

## 19. Current Empirical Status

This section is a running log of actual results from the repo, kept
separate from the forward-looking roadmap above. All results are 10-seed
means on controlled synthetic dynamical systems; full per-seed and
multi-seed tables are in `figures/*_multiseed_metrics.csv`.

### Stage 1: Detection results

- **Sine to quasi-periodic** (`experiments/synthetic_regime_shift.py`):
  TAMC (H1) AUROC 0.996 +/- 0.004, delay 11 steps, zero false alarms.
  Cleanest result so far; the source regime's clean periodic loop is
  exactly what H1 loop-tracking is suited to.
- **Logistic map, hardened** (`experiments/logistic_map_shift.py`,
  `r=3.45 -> r=3.75` with observation noise so the comparison isn't
  trivial): TAMC (H0) AUROC 0.9998 +/- 0.0007, delay 3.8 steps, zero false
  alarms — competitive with the strongest baselines (autocorrelation
  0.9989 AUROC/6.2 delay; spectral 0.9997 AUROC/3.8 delay) and clearly
  ahead of naive mean/variance drift (0.76/0.72 AUROC).
- **Lorenz** (`experiments/lorenz_shift.py`, stable equilibrium `rho=20`
  -> chaotic attractor `rho=28`): TAMC (H0) beats every baseline on every
  axis — AUROC 0.964 vs autocorrelation 0.942 and spectral 0.949; delay
  70.6 steps vs 113 and 117 respectively; zero false alarms; highest
  separation (6.33) of all methods.
- Across all three, the right homology dimension (H0 vs H1) tracked the
  *shape* of the source regime's attractor rather than a fixed default —
  see [methodology.md, Section 2](methodology.md#2-homology-dimension-choice-h0-vs-h1).

### Stage 2/3: TAMC-Lite forecasting/adaptation result

`experiments/tamc_lite_synthetic_forecast.py` (sine to quasi-periodic
shift, `LinearARForecaster` frozen baseline):

- **Simple residual adapters failed to net any benefit.**
  `MeanShiftResidual` actively hurt pre-shift accuracy (it reacts to
  within-cycle phase, not real drift) and never recovered post-shift.
  `AnalogResidualAdapter` (k-NN over source-regime residuals) was
  statistically indistinguishable from the frozen forecaster everywhere —
  it had no capacity to express a correction larger than the (tiny)
  in-sample fitting residual it was trained to memorize.
- **Topology-gated forecast blending is the strongest current adaptation
  result.** `TamicBlendPipeline` blends the frozen forecaster with a
  forward-only adaptive forecaster (`RecentPatternForecaster`) under a
  topological gate. 10-seed Net Adaptation Score: TAMC-gated blend
  **0.0257** (best of all variants) vs always-on 50/50 blend 0.0119,
  driven by preserving pre-shift accuracy (Pre Harm 0.0080 vs 0.0217)
  while matching the always-on blend's post-shift gain (~0.0336 either
  way).
- **Caveat:** `Post Improvement %` is noisy across seeds (mean 0.30%, std
  27% for the TAMC-gated blend) because it is a per-seed ratio with a
  denominator (frozen post-shift MAE) that varies a lot seed-to-seed.
  `Net Adaptation Score` should be the metric reported and compared, not
  `Post Improvement %` in isolation. See
  [methodology.md, Section 5](methodology.md#5-adaptation-tradeoff-metrics).

### Real-data controlled perturbation: ETTh1

- **Dataset:** ETTh1, `OT` column. **Local file:** `data/ETTh1.csv` (never
  downloaded by the script). **Experiment:**
  `experiments/real_data_controlled_shift.py`.
- **Shift types supported:** `amplitude`, `trend`, `noise`,
  `seasonality_break`, `frequency_proxy` — deterministic, controlled
  perturbations injected into the post-shift half only, after causal
  (source-half-only) normalization.
- **Main reported result:** `seasonality_break` (mixing the post-shift
  segment with its reverse-time copy, which inverts the phase of any
  recurrence structure). 10-seed Net Adaptation Score by gate:

  | Gate | Net Adaptation Score |
  |---|---|
  | TAMC-gated blend | **0.0027** |
  | Autocorrelation-gated | 0.0015 |
  | Spectral-gated | -0.0003 |
  | Mean/variance-gated | -0.0052 |
  | Always-on 50/50 blend | -0.0305 |
  | Adaptive recent-pattern alone | -0.1520 |

- **Interpretation:** TAMC has the best adaptation tradeoff of every gate
  tested on this real series, including the three non-topological gates
  that use the identical z-scored-sigmoid control law (see
  [methodology.md, Section 3](methodology.md#3-tamc-as-meta-control-not-an-additive-loss-term)),
  but the margin is modest — 0.0027 vs. autocorrelation's 0.0015 is not a
  large gap, unlike the much clearer separations seen in the synthetic
  detection experiments.
- **Caveat:** only the `noise` shift type is actually stochastic; the
  other four (`amplitude`, `trend`, `seasonality_break`, `frequency_proxy`)
  are fully deterministic given the input segment, so multi-seed runs on
  them correctly show zero std across seeds — this is expected behavior,
  not a sign the experiment didn't run.
- **Caveat:** this is a *controlled, injected* perturbation on real data,
  not a naturally occurring distribution shift. It establishes that the
  method still functions and still wins on real (non-synthetic) values and
  real (non-synthetic) noise characteristics, not that it would detect or
  usefully gate adaptation to a real-world regime change that arises on
  its own.

**Stochastic robustness check (`noise` shift, std 0.35, 10 seeds):** since
this is the only shift type that actually varies across seeds, it is the
honest test of whether the `seasonality_break` result is robust rather
than a lucky deterministic draw. 10-seed Net Adaptation Score (mean +/-
std):

| Gate | Net Adaptation Score |
|---|---|
| TAMC-gated blend | **0.0011 +/- 0.0015** |
| Autocorrelation-gated | -0.0015 +/- 0.0017 |
| Spectral-gated | -0.0018 +/- 0.0017 |
| Mean/variance-gated | -0.0034 +/- 0.0021 |
| Always-on 50/50 blend | -0.0394 +/- 0.0060 |
| Adaptive recent-pattern alone | -0.1853 +/- 0.0152 |

TAMC-gated blend has the highest mean and is the only gate with a
*positive* mean Net Adaptation Score under pure noise — every other gated
blend nets slightly harmful on average here. That said, this should not
be oversold as a clean win: TAMC's mean (0.0011) is smaller than its own
std (0.0015), so the result is not statistically distinguishable from
zero. The honest reading is that under pure Gaussian noise, with no real
structural shift for any drift signal to detect, no gate provides a
robust net benefit, and TAMC is the least-bad / most net-neutral option
rather than a clear winner. This is a genuine limitation, not a bug:
topology (like the other drift signals tested) has nothing structural to
latch onto when the "shift" is just added noise.

### Current limitation

Adaptation results so far establish that topology-gated *blending*
between two full forecasts works where topology-gated *residual
correction* did not, and that this holds on one real series (ETTh1) in
addition to the synthetic sine/quasi-periodic shift — though the real-data
margin over the best non-topological gate is modest. **Update:** the
logistic-map and Lorenz forecast-adaptation gap below this is now
resolved — see "Dynamical-system forecast adaptation" — with a mixed
result (clear win on Lorenz, no help on logistic map, sine result
nuanced). What remains untested is a *naturally occurring* real-world
shift, and a benchmark against the learned non-topological baselines
listed in Section 9 (DynaTTA-style embedding drift, COSA-style output
adapters, PETSA-style adapters) — the non-topological gates compared so
far (mean/variance, autocorrelation, spectral) are hand-rolled drift
scores under TAMC's own control law, not those papers' actual methods.
The forward-only adaptive forecasters themselves (`RecentPatternForecaster`,
`RollingLinearARForecaster`) are simple heuristics, not learned models,
and the dynamical-system results show this matters: **adapter inductive
bias is now the main adaptation bottleneck**, not the gating mechanism.
Topology decides *when* to trust an adaptive component; it cannot make a
mismatched adaptive component good. Net adaptation benefit depends
jointly on whether the shift is topologically detectable *and* on whether
the gated adaptive forecaster actually fits the post-shift dynamics.

### Next step

1. ~~Run the topology-gated blend on the logistic map and Lorenz shifts to
   check the result generalizes beyond the sine/quasi-periodic case.~~ Done
   -- see "Dynamical-system forecast adaptation" below. Result is mixed:
   generalizes (and wins) on Lorenz, fails to help on logistic map, and
   complicates rather than confirms the original sine-only story once a
   wider variant set is tried there.
2. Find or construct a naturally occurring real-world regime shift (not an
   injected perturbation) to test whether the controlled-perturbation
   result holds outside the controlled setting.
3. Replace `RecentPatternForecaster` with a stronger forward-only adaptive
   forecaster, and benchmark the topology-gated blend against the actual
   learned non-topological baselines listed in Section 9 (DynaTTA, COSA,
   PETSA), not just hand-rolled drift-score gates. (`RollingLinearARForecaster`,
   added for the dynamical-system experiment below, is a first step toward
   a stronger adaptive forecaster, but is itself unstable as a standalone
   forecaster on logistic map and Lorenz -- see below.)

### Dynamical-system forecast adaptation

`experiments/dynamical_forecast_adaptation.py` extends the forecast-
adaptation evidence (previously only sine/quasi-periodic and ETTh1) to the
two other controlled detection systems, asking directly: does
topology-gated adaptation generalize across multiple dynamical regime
shifts, or is the adaptation evidence mostly limited to the sine case?

- **Systems tested:** `sine_quasiperiodic`, `logistic_map`, `lorenz` (same
  causal generators as the detection experiments).
- **Variants compared (10 total):** frozen forecaster; two standalone
  adaptive forecasters (`RecentPatternForecaster`, and a new
  `RollingLinearARForecaster` that refits a small ridge-regularized AR
  model from the current context window alone at every prediction call,
  falling back to last-value repeat on too-short context); always-on and
  TAMC-gated 50/50 blends of frozen with each adaptive forecaster; and
  mean/variance-, autocorrelation-, and spectral-gated blends (under the
  same `ScalarDriftSignal` control law as `real_data_controlled_shift.py`)
  paired with `RollingLinearARForecaster` as the fixed "best adaptive
  forecaster" for the non-topological gates -- a decision made before
  looking at any post-shift result, per the experiment's design note,
  rather than a per-system validation search.
- **Topology setup:** H0 throughout (per the ablation's finding that H0 is
  the more robust default), with each system's existing detection delay
  (sine: 8, logistic map: 2, Lorenz: 6) and window=128.
- **10-seed Net Adaptation Score, key variants** (full table in
  `figures/dynamical_forecast_adaptation_tradeoff_summary.csv`):

  | System | TAMC-gated (recent-pattern) | Always-on (recent-pattern) | Best variant overall |
  |---|---|---|---|
  | Sine -> quasi-periodic | +0.0335 +/- 0.0982 | +0.0119 +/- 0.1162 | Adaptive rolling LinearAR alone (+0.2673 +/- 0.1546) |
  | Logistic map | -0.2855 +/- 0.0656 | -0.5211 +/- 0.0457 | Frozen forecaster (0.0000) |
  | Lorenz | **+0.1668 +/- 0.1144** | +0.1462 +/- 0.1222 | **TAMC-gated blend (recent-pattern)** |

- **Honest result -- mixed, not a clean generalization.**
  - **Lorenz: a genuine new win.** TAMC-gated blend (recent-pattern) is
    the best variant overall, beating always-on blending and all three
    non-topological gates (autocorrelation -0.7071, mean/variance -0.6407,
    spectral -0.6123, all strongly negative). This is real evidence beyond
    the sine-only result the limitation called out.
  - **Logistic map: nothing helps, and TAMC does not change that.** Every
    adaptive/blend variant is worse than the frozen forecaster alone,
    including TAMC-gated blending. TAMC-gated (recent-pattern) at -0.2855
    is the *least harmful* of every non-frozen variant (vs. always-on's
    -0.5211 and the non-topological gates' -1.43 to -1.44), so the gate is
    still doing its harm-minimization job correctly -- it just cannot
    manufacture a benefit where the frozen forecaster is already
    sufficient and no adaptive forecaster tried adds value.
  - **Sine: complicates rather than confirms the earlier story.** TAMC-
    gated blending is still positive (consistent with the earlier
    `tamc_lite_synthetic_forecast.py` result), but is no longer the best
    variant once `RollingLinearARForecaster` is added to the comparison:
    a *standalone*, ungated rolling-AR forecaster (+0.2673) and even
    autocorrelation-gated blending (+0.1622, paired with the same
    rolling-AR forecaster) both beat TAMC-gated blending here. The earlier
    sine result was correct given the variants it compared, but was not
    the full picture once a stronger adaptive forecaster is on the table.
  - **`RollingLinearARForecaster` alone is reliably harmful as a
    standalone forecaster** on every system (sine -0.1227 to -0.2337
    across the two runs, logistic map -2.52, Lorenz -4.67), despite
    sometimes contributing positively once blended (sine). It is not a
    safe drop-in replacement for `RecentPatternForecaster`.
- **Does this reduce the "adaptation evidence is sine/ETTh1-only"
  limitation?** Partially. It adds one genuine, robust new win (Lorenz)
  and one informative new failure mode (logistic map, where nothing
  helps), so the limitation should be sharpened rather than simply
  removed: TAMC-gated adaptation generalizes to *some* but not all
  dynamical regime shifts, and even on the system where the original
  result came from (sine), it is not uniformly the best choice once more
  adaptive-forecaster options are considered.
- 3-seed and 10-seed runs agree on every qualitative conclusion above
  (which system wins, which fails, which forecaster is unstable); only
  magnitudes shift slightly between them.

### Topology ablation status

`experiments/topology_ablation.py` is a dedicated ablation (not a
leaderboard) over three topological modeling choices, across all three
controlled detection systems used elsewhere in this repo:

- **What was swept:** homology dimension (H0 vs H1), embedding delay
  (default `2, 4, 6, 8, 12`), and topology window size (default `64, 128`;
  `192` is supported via `--windows` but excluded from the default grid to
  keep the default 10-seed run's persistent-homology call count
  manageable). Embedding dimension is held fixed at 3 by default. All
  three systems (`sine_quasiperiodic`, `logistic_map`, `lorenz`) use their
  existing causal generators and `detection_metrics` definitions
  unchanged — no system-specific tuning was introduced for this ablation.
- **Why:** Sections 2 of [methodology.md](methodology.md#2-homology-dimension-choice-h0-vs-h1)
  and this section's earlier detection results report H0-vs-H1 as a
  *post-hoc* observation from picking one dimension per system. This
  ablation runs the full grid honestly, so that claim can be checked
  systematically rather than asserted from three single data points.
- **Where outputs are saved:** `figures/topology_ablation_metrics.csv`
  (long-format, one row per system/seed/drift-dimension/delay/window),
  `figures/topology_ablation_summary.csv` (grouped mean/std over seeds),
  and `figures/topology_ablation_heatmap.png` (one AUROC-mean heatmap per
  system x drift-dimension, delay on the x-axis, window on the y-axis).
- **Status:** the full default grid (10 seeds x 3 systems x 2 dimensions x
  5 delays x 2 windows = 600 rows) ran to completion in 1227.8s (~20.5
  min). 10-seed AUROC mean +/- std, summarized from
  `figures/topology_ablation_summary.csv`:

  | System | H0 range | H1 range |
  |---|---|---|
  | sine_quasiperiodic | 0.9945-0.9984 (std <= 0.005) | 0.857-0.996 (one cell collapses) |
  | logistic_map | 0.9991-0.9998 (std <= 0.0025) | 0.984-0.996 (std up to 0.013) |
  | lorenz | 0.913-0.982 (std up to 0.13 at the worst cell) | 0.516-0.801 (std 0.20-0.33 throughout) |

  **Honest finding, and a correction to the earlier single-data-point
  claim:** the systematic sweep does *not* cleanly support "H1 is
  strongest for loop-like periodic/quasi-periodic attractors." H0 is
  uniformly excellent and low-variance across the *entire* grid for all
  three systems, including the two loop-like ones. H1 is competitive with
  H0 on most of the sine/logistic-map grid (sometimes a hair ahead on an
  individual cell) but is not clearly better anywhere, and it has a real
  failure mode H0 does not share: at `delay=12, window=64`, sine's H1
  AUROC collapses to 0.857 +/- 0.031 while H0 stays at 0.994 +/- 0.004 in
  that same cell. For Lorenz, the original claim holds strongly: H0
  (0.91-0.98) clearly and consistently beats H1 (0.52-0.80), and H1 is
  also far more unstable across seeds there (std up to 0.33, vs H0's
  worst-case 0.13). The revised, ablation-supported claim is: **H0 is the
  more robust default across all three systems and the full delay/window
  grid tested; H1 can match it on loop-like attractors but is never
  clearly better and has its own failure modes, and is unreliable on
  Lorenz's compact-to-chaotic transition.** This replaces the earlier,
  weaker claim in [methodology.md, Section 2](methodology.md#2-homology-dimension-choice-h0-vs-h1),
  which was based on one hand-picked dimension per system rather than a
  full sweep.

### Runtime benchmark status

`experiments/runtime_benchmark.py` measures the per-window compute cost of
TAMC's topological drift (H0 and H1) against three non-topological
baselines (mean/variance, autocorrelation, spectral), on the sine-to-
quasi-periodic series, at window sizes 64/128/192 (`--stride 8`,
`--max-windows 100`, `--n-repeat 3`). Persistence is computed once per
window for both H0 and H1 (not duplicated), but each row's reported cost
is the *full* standalone persistence cost plus that dimension's own
Wasserstein-distance call -- matching how `TamicSignal` is actually used
elsewhere in this repo (one fixed `drift_dimension`, no sharing across
dimensions), not an artificially halved "shared" cost. Outputs:
`figures/runtime_benchmark_metrics.csv` (per-repeat), `figures/runtime_benchmark_summary.csv`
(mean/std over repeats), `figures/runtime_benchmark.png` (log-scale
seconds-per-window vs. window size, one line per method).

**Result (full default run, completed in ~70s):**

| Window | TAMC H0 sec/window | TAMC H1 sec/window | Spectral sec/window | Autocorrelation sec/window | H0 vs. spectral | H0 vs. autocorrelation |
|---|---|---|---|---|---|---|
| 64 | 0.0049 | 0.0029 | 0.000013 | 0.000075 | 367.6x | 64.3x |
| 128 | 0.0413 | 0.0396 | 0.000014 | 0.000077 | 3045.1x | 536.6x |
| 192 | 0.1810 | 0.1746 | 0.000014 | 0.000078 | 13178.7x | 2320.0x |

**Honest reading:** the non-topological baselines are essentially flat
with window size (microseconds, dominated by fixed Python/NumPy overhead,
not by the O(window) work itself). TAMC's cost grows steeply and
roughly an order of magnitude per ~1.5x increase in window size, because
Vietoris-Rips persistent homology scales poorly with point-cloud size --
this is the real, dominant computational cost of TAMC, not an artifact of
this benchmark's implementation. At window=192, computing TAMC H0 drift
for one window takes ~181ms vs. ~14 microseconds for spectral drift: a
~13,000x slowdown. This means TAMC is impractical at large windows or high
update frequency without mitigation (sparser striding, smaller windows,
edge-collapse/approximate persistence, or computing topology only every
k steps -- see [Section 10, Risk 4](#risk-4-persistent-homology-is-expensive) above).
At window<=128 with `stride=8` (the defaults used throughout this repo's
detection experiments), the absolute cost is still small (tens of
milliseconds per window), so the existing experiments remain practical,
but this would not scale to windows in the many hundreds or to per-step
(stride=1) scoring without further optimization.