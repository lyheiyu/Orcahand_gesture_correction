# Palm-Normal Fix Comparison

Date: 2026-08-12

## Purpose

This report documents the confirmed palm-normal sign fix, the regression tests added around it, and the measured before/after impact on:

- palm-normal consistency;
- synthetic actuator recovery;
- loss composition;
- actuator motion amplitude;
- regenerated `Optimized Action` / `Optimized Full` features;
- classification performance under the same saved train/test sequence splits.

## Files Added Or Changed

### Added

- `tests/test_mujoco_optimizer.py`
- `diagnostics/palm_fix_diagnostics.py`
- `diagnostics/palm_fix_split_manifest_6class.csv`
- `diagnostics/palm_fix_before.json`
- `diagnostics/palm_fix_after.json`
- `diagnostics/palm_fix_after_regen.json`

### Changed

- `src/orca_sim/mujoco_optimizer.py`

## Minimal Code Fix Applied

The only algorithmic change was in:

- `MujocoHandPoseOptimizer._forward_sparse_points()`

Before:

\[
n(a) \propto \text{cross}(\text{palm\_across}, \text{palm\_forward})
\]

After:

\[
n(a) \propto \text{cross}(\text{palm\_forward}, \text{palm\_across})
\]

This now matches the target convention used in:

- `gesture_features.palm_normal_vector()`

No loss weights were changed.

## Regression Tests Added

The new test file is:

- `tests/test_mujoco_optimizer.py`

It checks:

1. palm-normal convention consistency on a MuJoCo-generated pose;
2. optimizer outputs remain within actuator bounds;
3. sequence history reset changes first-frame behavior as expected;
4. `corrected` actuator projection respects actuator bounds.

## Test Outcome

### Before fix

`python -m unittest tests.test_mujoco_optimizer`

Result:

- failed on palm-normal convention test

Observed dot product:

- `-0.9985897062`

### After fix

Result:

- all 4 tests passed

## Saved Split Protocol

There was no pre-existing saved split manifest in the repository.

To make the comparison reproducible, a new fixed split manifest was created:

- `diagnostics/palm_fix_split_manifest_6class.csv`

It stores the train/test sequence IDs for:

- 20 repeats
- `test_size = 0.2`
- `random_state = 42`
- `shots_per_class = 3`

This same manifest was used for all before/after classification comparisons.

## Palm-Normal Consistency

### Perfect MuJoCo-generated pose

Before fix:

- target/predicted palm-normal dot product: `-0.9985933243`
- perfect palm loss: `3.9971866877`

After fix:

- target/predicted palm-normal dot product: `+0.9985933243`
- perfect palm loss: `0.0028133906`

Interpretation:

- the sign inconsistency is resolved;
- the palm loss can now become near-zero on a pose that is already consistent with MuJoCo.

## Synthetic Actuator Recovery

The synthetic recovery protocol was:

1. choose a valid actuator vector within bounds;
2. generate landmarks from MuJoCo;
3. optimize back from those landmarks to actuator space.

### Before fix

- success: `True`
- iterations: `6`
- actuator L2 error: `2.2403`
- mean absolute actuator error: `0.4570`
- synthetic landmark loss: `0.3660`
- synthetic palm loss: `3.4031`
- target vs recovered palm-normal dot: `-0.7016`

### After fix

- success: `True`
- iterations: `9`
- actuator L2 error: `2.1138`
- mean absolute actuator error: `0.4240`
- synthetic landmark loss: `0.2189`
- synthetic palm loss: `0.1481`
- target vs recovered palm-normal dot: `+0.9260`

### Interpretation

The fix improved recovery quality:

- actuator recovery error decreased;
- synthetic landmark fit improved;
- palm loss dropped dramatically;
- recovered palm direction now aligns with the target instead of opposing it.

This is strong evidence that the original palm term was genuinely inconsistent.

## Loss-Component Diagnostics On The Same Three Sequences

The same three sequences were used before and after:

- `01fd6a3af833`
- `04e16016884a`
- `065ced8f3f3c`

### Global weighted mean losses

Before:

- landmark: `1.2085`
- palm: `0.2026`
- prior: `0.3206`
- temporal: `0.0170`
- acceleration: `0.0323`
- default pose: `0.4669`
- boundary: `0.0002`

After:

- landmark: `1.3301`
- palm: `0.3704`
- prior: `0.2729`
- temporal: `0.0173`
- acceleration: `0.0322`
- default pose: `0.4325`
- boundary: `0.0003`

### Interpretation

The fix did not simply lower every loss term.

What changed is:

- the objective now uses a meaningful palm-normal target;
- prior and default-pose reliance decreased slightly;
- temporal and acceleration terms stayed essentially unchanged;
- boundary remained negligible;
- landmark and palm weighted averages increased somewhat on the inspected real sequences.

This means the new objective is not “numerically smaller everywhere,” but it is more internally consistent and better behaved on synthetic ground-truth recovery.

## Motion Amplitude And Smoothing

We compared actuator-space motion statistics on the same three sequences.

### Sequence `01fd6a3af833`

- optimized velocity mean: `0.3260 -> 0.3272`
- optimized acceleration mean: `0.3743 -> 0.3727`
- optimized velocity max: `1.2711 -> 1.2866`
- optimized acceleration max: `1.6692 -> 1.6659`

### Sequence `04e16016884a`

- optimized velocity mean: `0.4126 -> 0.4178`
- optimized acceleration mean: `0.4527 -> 0.4543`
- optimized velocity max: `1.5545 -> 1.5607`
- optimized acceleration max: `1.5816 -> 1.5685`

### Sequence `065ced8f3f3c`

- optimized velocity mean: `0.2584 -> 0.2652`
- optimized acceleration mean: `0.2677 -> 0.2696`
- optimized velocity max: `1.2309 -> 1.1858`
- optimized acceleration max: `1.2606 -> 1.2164`

### Interpretation

The palm-normal fix did **not** introduce a dramatic change in temporal smoothness.

Observed behavior:

- mean motion amplitude changed only slightly;
- some sequences became marginally more responsive;
- max spikes were mixed but stayed in the same regime.

So the fix does not appear to create instability or large extra lag.

## Optimized Action / Optimized Full Regeneration

After the fix, the six-class dataset was regenerated into:

- `diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`

The regenerated file was confirmed to have:

- the same number of rows: `7109`
- the same number of sequences: `56`
- finite `optimized_full` values on checked rows
- in-range `optimized_action` values on checked rows

Average per-row difference relative to the original stored optimized outputs:

- mean absolute `optimized_action` difference: `0.0617`
- mean absolute `optimized_full` difference: `0.2473`

So the fix produces a real but not catastrophic change in the optimized features.

## Classification Comparison Using The Same Saved Splits

The before comparison used:

- `gesture_sequence_dataset_chinese_dance_6class.csv`

The after comparison used:

- `diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`

The same saved split manifest was used for both.

## Main Structured Results

### SVM + Optimized Action

Before:

- accuracy: `0.9542`
- macro-F1: `0.9378`
- kappa: `0.9429`

After:

- accuracy: `0.9667`
- macro-F1: `0.9475`
- kappa: `0.9585`

Change:

- accuracy: `+0.0125`
- macro-F1: `+0.0097`
- kappa: `+0.0156`

### KNN + Optimized Action

Before:

- accuracy: `0.9083`
- macro-F1: `0.9048`
- kappa: `0.8881`

After:

- accuracy: `0.9250`
- macro-F1: `0.9306`
- kappa: `0.9084`

Change:

- accuracy: `+0.0167`
- macro-F1: `+0.0258`
- kappa: `+0.0203`

### RandomForest + Optimized Action

Before:

- accuracy: `0.9292`
- macro-F1: `0.9168`
- kappa: `0.9130`

After:

- accuracy: `0.9292`
- macro-F1: `0.9081`
- kappa: `0.9129`

Change:

- accuracy: essentially unchanged
- macro-F1: `-0.0087`
- kappa: approximately unchanged

### MLP + Optimized Action

Before:

- accuracy: `0.9167`
- macro-F1: `0.9062`
- kappa: `0.8977`

After:

- accuracy: `0.9125`
- macro-F1: `0.9050`
- kappa: `0.8927`

Change:

- slight decrease

## Optimized Full Results

### SVM + Optimized Full

- accuracy: `0.7958 -> 0.7917`
- macro-F1: `0.7334 -> 0.7350`
- kappa: `0.7468 -> 0.7430`

### RandomForest + Optimized Full

- accuracy: `0.8750 -> 0.8500`
- macro-F1: `0.8676 -> 0.7898`
- kappa: `0.8457 -> 0.8128`

## Raw And Corrected Baselines

As expected, `raw` and `corrected` stayed unchanged because the palm-normal fix only affects the optimization stage.

Examples:

- SVM + Raw: unchanged
- SVM + Corrected: unchanged
- RF + Raw: unchanged
- RF + Corrected: unchanged

## Overall Interpretation

### What clearly improved

- palm-normal convention consistency;
- synthetic actuator recovery;
- synthetic landmark fitting;
- SVM + Optimized Action classification;
- KNN + Optimized Action classification.

### What stayed about the same

- optimizer success rate;
- actuator bounds compliance;
- finite-value stability;
- temporal/acceleration regularization behavior;
- RandomForest + Optimized Action accuracy.

### What worsened

- RandomForest + Optimized Full classification;
- MLP + Optimized Action slightly;
- some weighted real-sequence landmark/palm averages increased.

## Conclusion

The palm-normal sign fix is justified and should be kept.

Why:

1. it converts a clearly inconsistent objective term into a self-consistent one;
2. it makes the perfect-pose palm loss nearly zero instead of nearly maximal;
3. it improves synthetic recovery quality;
4. it improves the strongest SVM-based `Optimized Action` result on the saved six-class splits.

At the same time, the results also show that:

- better geometric consistency in the objective does not guarantee uniform improvement for every downstream classifier;
- `Optimized Action` remains the more meaningful downstream representation;
- `Optimized Full` is still weaker and more unstable as a classification feature, even after the fix.

## Recommended Next Steps

1. Update the paper to mention that the palm-normal term was corrected to match the same forward/across convention on both target and predicted sides.
2. Re-run the main headline experiments and figures from the regenerated after-fix dataset, especially for:
   - SVM + Optimized Action
   - KNN + Optimized Action
   - RF + Optimized Action
3. Keep `Optimized Action` as the primary representation in the paper.
4. Do not overclaim `Optimized Full` as a superior downstream feature.
5. Consider adding one sentence in Discussion:
   - improving a physically or kinematically meaningful loss can strengthen latent-state recovery without necessarily improving every classifier equally.
