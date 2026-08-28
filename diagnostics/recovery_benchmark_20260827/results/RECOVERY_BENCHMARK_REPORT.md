# Controlled Clean-Reference Recovery Benchmark

## Scope

This benchmark treats the recorded six-class MediaPipe trajectories as a controlled clean reference, not as physical ground truth. The current fixed-weight Optimized Action implementation and all filter parameters were left unchanged.

Each sequence receives one balanced corruption condition. Random seeds rotate through a deterministic pool of ten values. Sequence/corruption instances, rather than frames, are the statistical units.

Two recovery definitions are reported:

- `direct_recovery_ratio = 1 - error(method(corrupt), clean_reference) / error(corrupt, clean_reference)` includes each method's clean-domain/model bias.
- `robust_recovery_ratio = 1 - error(method(corrupt), method(clean)) / error(corrupt, clean_reference)` isolates sensitivity to the induced corruption.

Corrected and Fixed OA actuator errors use each method's clean output as a controlled reference; they are not errors against measured joint-angle ground truth.
Across all conditions, normalized actuator sensitivity is `0.0182` for Fixed OA and `0.0292` for Corrected. For Gaussian noise it is `0.0605` versus `0.1020`.

## Main Results

| Method | Clean-domain landmark bias | Corruption sensitivity in affected region | Robust recovery ratio | Velocity sensitivity | Acceleration sensitivity | Median amplitude retention |
|---|---:|---:|---:|---:|---:|---:|
| Corrupted Input | 0.0000 | 0.3293 | 0.0000 | 0.0217 | 0.0379 | 1.0236 |
| Kalman | 0.2208 | 0.0712 | 0.7399 | 0.0031 | 0.0039 | 1.0133 |
| One-Euro | 0.1030 | 0.1817 | 0.4512 | 0.0079 | 0.0111 | 1.0100 |
| Corrected | 1.6887 | 0.2939 | -0.2517 | 0.0411 | 0.0715 | 1.1071 |
| Fixed OA | 1.7049 | 0.1979 | 0.1910 | 0.0210 | 0.0291 | 1.1029 |

Kalman and One-Euro provide the strongest landmark-space robustness in this controlled benchmark. Fixed OA is substantially less sensitive than Corrected, but it does not outperform the conventional filters. The large clean-domain bias of the ORCA reconstructions means that `Optimized Full` is not currently aligned closely enough with the MediaPipe clean reference to support a direct landmark-correction claim.

Within actuator space, however, Fixed OA consistently reduces corruption sensitivity relative to Corrected:

| Corruption | Corrected actuator MAE | Fixed OA actuator MAE | Relative reduction |
|---|---:|---:|---:|
| All | 0.0292 | 0.0182 | 37.6% |
| Gaussian | 0.1020 | 0.0605 | 40.7% |
| Spike | 0.0060 | 0.0050 | 17.8% |
| Dropout | 0.0059 | 0.0046 | 21.9% |

The overall actuator-MAE difference is significant under a paired sequence-level Wilcoxon test (`p = 7.97e-94`). The actuator velocity and acceleration differences are also significant (`p = 3.24e-95` for both). These tests quantify robustness of the latent representation; they do not establish joint-angle accuracy against physical ground truth.

## Decision Questions

### Q1. Does Fixed OA reduce induced trajectory error?

No under direct clean-reference landmark error because the ORCA reconstruction has a large clean-domain bias. Yes under the bias-adjusted sensitivity criterion: the mean robust recovery ratio is `0.191`. This is below Kalman (`0.740`) and One-Euro (`0.451`).

### Q2. Does Fixed OA outperform Corrected for Gaussian noise?

Not in absolute reconstructed landmark error (`1.7214` versus `1.6940`). It is less corruption-sensitive than Corrected in both reconstructed landmark space (robust ratios `-0.333` versus `-1.066`) and actuator space (MAE `0.0605` versus `0.1020`). Both ORCA representations still amplify Gaussian perturbations in landmark space.

### Q3. Does Fixed OA outperform Corrected for isolated spikes?

Not in absolute reconstructed landmark error (`1.9816` versus `1.8958`), but yes in bias-adjusted landmark sensitivity (robust ratios `0.486` versus `0.241`) and actuator sensitivity (`0.0050` versus `0.0060`). Fixed OA is approximately equal to One-Euro (`0.485`) for spike robust ratio, while Kalman remains stronger (`0.841`).

### Q4. Does Fixed OA outperform Corrected for 3-frame and 5-frame dropout?

Absolute reconstructed errors are mixed: OA is marginally lower for 3 frames (`2.0040` versus `2.0119`) and higher for 5 frames (`1.9795` versus `1.9545`). Bias-adjusted OA robust ratios are positive for both durations (`0.333` and `0.255`), whereas Corrected is negative (`-0.129` and `-0.088`). Kalman and One-Euro remain stronger.

### Q5. Does Fixed OA preserve motion amplitude?

Its mean sequence-level median own-baseline retention is `1.103` (ideal `1.0`). This does not indicate systematic motion collapse, although Gaussian corruption can increase latent amplitude and should not be described as harmless smoothing.

### Q6. Is Fixed OA recovering the trajectory or mainly smoothing it?

Fixed OA is less sensitive than Corrected and the unprocessed trajectory in aggregate, particularly for isolated spikes. However, conventional Kalman and One-Euro filters recover the clean landmark trajectories more effectively, and Gaussian perturbations remain problematic. The present result supports a model-constrained robust actuator representation, not verified landmark recovery or superior general-purpose denoising.

## Decision

This benchmark most closely matches Outcome B, with limited spike-related evidence from Outcome C. The fixed-weight method improves the stability of the latent actuator representation, but it does not establish superior clean-reference landmark recovery. Before retaining `landmark correction` as the paper's central claim, the coordinate/reconstruction bias and observation model need to be addressed. If the first paper remains unchanged, its contribution should be framed around robust structured representation and downstream utility rather than recovered landmark accuracy.

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
