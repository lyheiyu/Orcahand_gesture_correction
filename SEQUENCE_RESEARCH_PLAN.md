# Sequence Research Plan

## Goal

Build a research pipeline that upgrades the current frame-wise MediaPipe-to-ORCA correction into a temporally consistent sequence correction method, and evaluate whether sequence-aware corrected features improve gesture recognition compared with raw MediaPipe landmarks.

The core hypothesis is:

> MediaPipe landmarks are noisy on a per-frame basis, while ORCA-constrained corrected features become more useful when temporal continuity is explicitly modeled across consecutive frames.


## Current Baseline

The repository already supports a frame-wise pipeline:

1. `MediaPipe landmarks`
2. `raw / geom / corrected` feature extraction
3. `few-shot SVM` classification

Current interpretation:

- `raw`: normalized 21x3 landmarks
- `geom`: hand-crafted geometric features
- `corrected`: ORCA-aware projected actuator-space features

This gives a strong single-frame baseline and should be kept as the reference system in the paper.


## Why Sequence Modeling

Single-frame correction has clear limitations:

- landmark jitter
- transient hand-tracking failures
- palm orientation ambiguity
- frame-wise inconsistency
- loss of motion continuity

Sequence modeling can add:

- temporal smoothness
- motion continuity priors
- short-term occlusion robustness
- stable correction trajectories


## Proposed Research Stages

### Stage 1: Sequence Dataset

Upgrade data collection from isolated saved frames to labeled short clips.

Each sample should include:

- `sequence_id`
- `frame_id`
- `label`
- timestamp or relative frame index
- `raw_*`
- `geom_*`
- `corrected_*`

Recommended collection protocol:

- 5 to 10 gesture classes
- 20 to 50 short clips per class
- each clip length: 20 to 60 frames
- include viewpoint variation
- include mild wrist rotation and natural motion variation
- optionally include multiple recording sessions

Recommended split:

- train / val / test by `sequence_id`
- later strengthen to `session_id`


### Stage 2: Simple Temporal Baselines

Before deep models, add simple temporal aggregation baselines.

Recommended baselines:

1. Frame-wise majority vote
   - classify every frame independently
   - vote across the clip

2. Sliding-window mean features
   - average features across `T` frames
   - train the same classifier on pooled features

3. Exponential smoothing
   - smooth `corrected` features over time
   - compare with unsmoothed features

This stage answers:

> Does temporal smoothing alone already improve recognition?


### Stage 3: Sequence-Aware Corrected Features

Add temporal correction on top of the current ORCA-aware projection.

Minimal sequence correction rule:

`corrected_t = alpha * corrected_current + (1 - alpha) * corrected_{t-1}`

Then extend with penalties such as:

- velocity penalty
- acceleration penalty
- palm-orientation continuity
- missing-frame interpolation

Interpretation:

- current frame gives observation
- previous corrected state gives temporal prior
- corrected output becomes a stable trajectory rather than isolated predictions


### Stage 4: Sequence Models

After temporal baselines are working, compare lightweight sequence models.

Recommended order:

1. `KNN` / `SVM` on pooled sequence statistics
2. `MLP` on concatenated short-window features
3. `Temporal CNN`
4. `LSTM` or `GRU`

Suggested feature inputs:

- raw sequence
- geom sequence
- corrected sequence
- hybrid sequence

Suggested sequence summary features for traditional models:

- mean over time
- std over time
- max over time
- start-end delta
- first-order velocity statistics


## Research Questions

The paper can be organized around these questions:

1. Do ORCA-aware corrected features improve few-shot frame-wise classification over raw MediaPipe landmarks?
2. Does temporal smoothing further improve corrected features?
3. Do sequence-aware corrected features outperform frame-wise corrected features?
4. Which classifiers benefit most from corrected sequence representations?


## Suggested Experiments

### Experiment A: Single-Frame Baseline

Compare:

- raw
- geom
- corrected
- hybrid

Models:

- KNN
- SVM
- MLP
- RandomForest

Metrics:

- accuracy
- macro F1
- precision
- recall
- confusion matrix
- mean +- std over repeated runs


### Experiment B: Few-Shot Sequence Classification

Compare training with:

- 1-shot
- 3-shot
- 5-shot

Evaluation units:

- per-frame accuracy
- per-sequence accuracy

This is especially important because corrected features may help more in low-data settings.


### Experiment C: Temporal Ablation

Compare:

- frame-wise corrected
- smoothed corrected
- short-window pooled corrected
- sequence-model corrected

This isolates the contribution of temporal information.


### Experiment D: Cross-Session Generalization

If possible, collect multiple sessions and split by session.

This tests whether the method generalizes beyond one recording condition.


## Mathematical Framing

Current frame-wise method:

- input: landmark frame `x_t`
- output: corrected feature `z_t`

Sequence-aware method:

- input: landmark sequence `x_1, ..., x_T`
- corrected sequence `z_1, ..., z_T`

Conceptually:

- `z_t` should stay close to the observation-derived ORCA projection
- `z_t` should also stay temporally consistent with `z_{t-1}`

A simple objective can be described as:

`L_t = L_obs(z_t, x_t) + lambda_s * L_smooth(z_t, z_{t-1})`

Possible extensions:

- `L_vel`
- `L_acc`
- `L_palm`

Important wording:

- prefer "structure-aware" or "ORCA-constrained"
- avoid claiming full physical optimization unless true optimization over the MuJoCo state is implemented


## Paper Positioning

Good current wording:

- ORCA-aware structural projection
- embodiment-constrained feature correction
- temporally consistent ORCA-constrained sequence features

Avoid overstating as:

- full physics-based reconstruction
- physically exact hand pose recovery

unless the optimization layer is actually implemented.


## Immediate Next Steps

1. Extend the collector to save short labeled sequences instead of isolated frames.
2. Add sequence IDs and frame IDs to the dataset format.
3. Implement a temporal smoothing baseline for `corrected`.
4. Add pooled sequence features for `KNN / SVM / MLP`.
5. Compare frame-wise and sequence-wise corrected features under few-shot settings.


## Minimal Publishable Result

A realistic first publishable contribution would be:

- frame-wise ORCA-aware corrected features improve few-shot gesture recognition over raw MediaPipe landmarks
- temporally smoothed corrected sequence features further improve robustness
- improvements are shown consistently across multiple classifiers and repeated runs

This would already be a credible research story for a small conference or workshop paper.
