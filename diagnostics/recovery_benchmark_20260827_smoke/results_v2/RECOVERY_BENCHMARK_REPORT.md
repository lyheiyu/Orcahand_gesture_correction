# Controlled Clean-Reference Recovery Benchmark

## Scope

This benchmark treats the recorded six-class MediaPipe trajectories as a controlled clean reference, not as physical ground truth. The current fixed-weight Optimized Action implementation and all filter parameters were left unchanged.

Each sequence receives one balanced corruption condition. Random seeds rotate through a deterministic pool of ten values. Sequence/corruption instances, rather than frames, are the statistical units.

Two recovery definitions are reported:

- `direct_recovery_ratio = 1 - error(method(corrupt), clean_reference) / error(corrupt, clean_reference)` includes each method's clean-domain/model bias.
- `robust_recovery_ratio = 1 - error(method(corrupt), method(clean)) / error(corrupt, clean_reference)` isolates sensitivity to the induced corruption.

Corrected and Fixed OA actuator errors use each method's clean output as a controlled reference; they are not errors against measured joint-angle ground truth.
Across all conditions, normalized actuator sensitivity is `0.0328` for Fixed OA and `0.0568` for Corrected. For Gaussian noise it is `0.0534` versus `0.0933`.

## Decision Questions

### Q1. Does Fixed OA reduce induced trajectory error? No by the direct corrupted-region criterion.
Mean direct recovery ratio: `-31.167`. Mean bias-adjusted robust recovery ratio: `-0.131`.

### Q2. Does Fixed OA outperform Corrected for Gaussian noise? No.
Corrupted-region error: Fixed OA `1.6515`, Corrected `1.6455`.

### Q3. Does Fixed OA outperform Corrected for isolated spikes? No.
Corrupted-region error: Fixed OA `1.7729`, Corrected `1.4507`.

### Q4. Does Fixed OA outperform Corrected for 3-frame and 5-frame dropout? 3 frames: No; 5 frames: No.
Errors (OA vs Corrected): 3 frames `nan` vs `nan`; 5 frames `nan` vs `nan`.

### Q5. Does Fixed OA preserve motion amplitude? Its mean sequence-level median own-baseline retention is `1.019` (ideal `1.0`).

### Q6. Is Fixed OA recovering the trajectory or mainly smoothing it?

The current evidence does not support a trajectory-recovery claim. Any reduction in derivative magnitude should be interpreted as smoothing unless the objective or observation model is revised.

## Validation

- Non-finite values: `0`
- Actuator bound violations: `0`
- Mean OA optimizer success rate: `1.0000`
- Sequence-start reset maximum action difference: `0.000e+00`

## Output Files

- `per_sequence_landmark_metrics.csv`: primary paired landmark metrics.
- `per_sequence_actuator_metrics.csv`: within-method actuator corruption sensitivity.
- `actuator_overall_summary.csv` and `actuator_summary_by_corruption.csv`: actuator-space aggregate results.
- `overall_summary.csv`, `summary_by_corruption.csv`, and `summary_by_condition.csv`: aggregate tables.
- `paired_wilcoxon_tests.csv`: paired sequence-level tests.
- `figures/`: recovery, motion-fidelity, and systematically selected trajectory plots.

## Interpretation Constraint

This experiment evaluates recovery toward an observed clean MediaPipe trajectory. It does not establish anatomical ground-truth accuracy, long-duration occlusion completion, or cross-subject generalization.
