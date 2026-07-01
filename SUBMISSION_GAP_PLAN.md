# Submission Gap Plan

This file turns the current feedback into a concrete pre-submission checklist.

## Overall Judgment

The paper already has a publishable core idea, but it is still closer to a strong full draft than to a stable submission version.

Current strengths:

- ORCA actuator space replaces direct use of `63D` MediaPipe landmarks
- MuJoCo forward kinematics introduces structure-aware constraints
- optimization includes robust fitting, first-order temporal smoothing, and second-order acceleration regularization
- `optimized_action` is clearly stronger than `raw` and `corrected` in the current six-class Chinese dance task
- the draft already reports a strong but honest competing result from `raw_pca17`

Current weakness:

- the experimental loop is not fully closed yet
- some claimed baselines appear in the writing but not yet in a complete final table
- dataset protocol and implementation detail are still under-specified

## Priority Order

### Priority 1: Close the experimental loop

1. Put smoothing baselines into one formal result table
2. Report PCA comparisons under the same classifier protocol
3. Report `mean ± std` consistently
4. Clarify dataset split protocol and class counts

### Priority 2: Improve reproducibility

5. Add a 17-actuator definition table
6. Add optimization implementation details

### Priority 3: Tighten the paper claim

7. Consider a safer title than direct `Landmark Correction`

## 1. Smoothing Baselines Must Enter the Main Results

The draft already mentions:

- Moving Average
- Savitzky-Golay
- One-Euro
- Kalman

These should appear in a formal table, not only in narrative text.

### Required Table Shape

| Representation | Accuracy | Macro-F1 | Kappa | Velocity RMS | Acceleration RMS |
|---|---:|---:|---:|---:|---:|
| Raw |  |  |  |  |  |
| Moving Average |  |  |  |  |  |
| Savitzky-Golay |  |  |  |  |  |
| One-Euro |  |  |  |  |  |
| Kalman |  |  |  |  |  |
| Corrected |  |  |  |  |  |
| Optimized Action |  |  |  |  |  |

### Current Available Numbers

These numbers are already available for the earlier three-class setting from `figures/paper_summary/`.

| Method | Accuracy | Macro-F1 | Kappa |
|---|---:|---:|---:|
| Raw | 0.5938 | 0.5418 | 0.3688 |
| Moving Average | 0.6062 | 0.5577 | 0.3893 |
| Savitzky-Golay | 0.6125 | 0.5671 | 0.3970 |
| One-Euro | 0.6375 | 0.5897 | 0.4365 |
| Kalman | 0.6250 | 0.5815 | 0.4190 |
| Optimized Full | 0.7625 | 0.7480 | 0.6371 |

These are useful as prior evidence, but they should be rerun or clearly separated if the main paper now uses the six-class Chinese dance subset.

## 2. PCA Must Be Reported Under the Same Comparison Logic

At the moment, the key comparison is not yet fully symmetric.

The minimum fair comparison is:

- `raw`
- `raw_pca17`
- `corrected`
- `optimized_action`
- `optimized_full`

for:

- `SVM`
- `KNN`
- `RandomForest`
- `MLP`

### Required Table Shape

| Classifier | Raw | PCA-17 | Corrected | Optimized Action | Optimized Full |
|---|---:|---:|---:|---:|---:|
| SVM |  |  |  |  |  |
| KNN |  |  |  |  |  |
| RF |  |  |  |  |  |
| MLP |  |  |  |  |  |

### Current Confirmed Six-Class Highlights

Structured representation:

- `SVM + optimized_action = 0.9542 accuracy`
- `RF + optimized_action = 0.9292 accuracy`
- `MLP + optimized_action = 0.9167 accuracy`

Strong PCA baseline:

- `RF + raw_pca17 = 0.9458 accuracy`

This is already enough to support the main story, but not yet enough for a fully fair classifier-wide table.

## 3. Mean ± Standard Deviation Must Be Unified

At minimum, every main result should be reported as:

- `accuracy_mean ± accuracy_std`
- `macro_f1_mean ± macro_f1_std`
- `kappa_mean ± kappa_std`

If time allows, also add:

- `95% confidence interval`
- `paired t-test` or `Wilcoxon signed-rank test`
- `effect size`

### Why This Matters

The gap between:

- `0.9542`
- `0.9458`

is small enough that significance may matter.

Without a variance estimate, the argument is weaker than it looks.

## 4. Dataset Description Must Be Tightened

This is one of the most likely review points.

The final paper should explicitly state:

- total number of sequences
- total number of frames
- per-class sequence counts
- number of participants
- whether all data come from one performer or multiple performers
- whether train and test contain the same participant
- whether the split is subject-independent
- whether multiple sequences were cut from the same source recording
- whether there is possible session leakage

### Required Table Shape

| Item | Value |
|---|---|
| Total sequences |  |
| Total frames |  |
| Number of classes | 6 |
| Participants |  |
| Sessions |  |
| Split protocol |  |
| Shots per class | 3 |
| Repeats | 20 |

### Per-Class Count Table

| Label | Sequence Count | Frame Count |
|---|---:|---:|
| orchid_palm |  |  |
| orchid_finger |  |  |
| flower_pinch |  |  |
| prayer_beads |  |  |
| three_finger_bent |  |  |
| deer_horn |  |  |

If this is mainly a one-performer or limited-performer dataset, the claim should be constrained accordingly.

Recommended wording:

`fine-grained within-domain Chinese dance gesture recognition`

## 5. Add a 17-Actuator Definition Table

At present, the paper says `17D actuator state`, but that is not yet reproducible enough.

### Required Table Shape

| Actuator | Derived Landmarks | Meaning | Range |
|---|---|---|---|
| wrist_yaw |  |  |  |
| wrist_pitch |  |  |  |
| wrist_roll |  |  |  |
| thumb_open |  |  |  |
| index_flexion |  |  |  |
| index_abduction |  |  |  |
| middle_flexion |  |  |  |
| ring_flexion |  |  |  |
| pinky_flexion |  |  |  |

The exact naming can follow your implementation, but the paper should make the mapping readable to another researcher.

## 6. Add Optimization Details

The current draft still needs an implementation paragraph or table that includes:

- optimizer name
- initialization source
- maximum iterations per frame
- Huber delta
- all lambda weights
- actuator bounds
- termination criterion
- runtime per frame
- CPU / GPU environment

### Required Table Shape

| Item | Value |
|---|---|
| Optimizer |  |
| Initialization | corrected / previous frame / etc. |
| Max iterations per frame |  |
| Huber delta |  |
| Temporal weight |  |
| Acceleration weight |  |
| Prior weight |  |
| Boundary weight |  |
| Runtime per frame |  |
| Hardware |  |

## 7. Consider a Safer Title

The current strongest output is:

- `optimized_action`

not:

- `optimized_full`

This means the strongest contribution is more clearly a **refined latent hand representation** than a fully validated landmark correction result.

### Safer Title Options

1. `MuJoCo-Constrained Temporal Refinement of Hand Landmark Sequences for Few-Shot Gesture Recognition`
2. `Robust Hand Pose Representation via MuJoCo-Constrained Temporal Optimization`
3. `MuJoCo-Constrained Refinement of Noisy Hand Landmark Sequences for Few-Shot Gesture Recognition`

## Recommended Immediate Action

If only one thing is done next, do this first:

### Build one unified main result table

For the six-class Chinese dance subset, create a single table covering:

- Raw
- Moving Average
- Savitzky-Golay
- One-Euro
- Kalman
- Corrected
- Optimized Action
- PCA-17

with:

- Accuracy
- Macro-F1
- Kappa
- Velocity RMS
- Acceleration RMS

This one table will close a large part of the current reviewer risk.

## Proposed Next Deliverables

1. `dataset_summary_6class.csv`
2. `main_results_unified_6class.csv`
3. `pca_classifier_matrix_6class.csv`
4. `actuator_definition_table.md`
5. `optimization_settings_table.md`

## Practical Note

The paper is not blocked by lack of a research story.

It is blocked by lack of a complete experimental package.

That is a much better problem to have.
