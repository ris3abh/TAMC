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

### Current limitation

Adaptation results so far establish that topology-gated *blending*
between two full forecasts works where topology-gated *residual
correction* did not, on one synthetic shift. This has not yet been tested
on the logistic map or Lorenz shifts, on real data, or against any of the
non-topological baselines listed in Section 9 (DynaTTA-style embedding
drift, COSA-style output adapters, PETSA-style adapters). The forward-only
adaptive forecaster itself (`RecentPatternForecaster`) is a simple,
autocorrelation-lag-based heuristic, not a learned model.

### Next step

1. Run the topology-gated blend on the logistic map and Lorenz shifts to
   check the result generalizes beyond the sine/quasi-periodic case.
2. Move to real-data controlled perturbations (DynaTTA/TTFBench-style
   trend, seasonality, and regime-shift injections on real series) rather
   than only fully-synthetic dynamical systems.
3. Replace `RecentPatternForecaster` with a stronger forward-only adaptive
   forecaster, and benchmark the topology-gated blend against the
   non-topological gates listed in Section 9.