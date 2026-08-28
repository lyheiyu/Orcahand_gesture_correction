# Final Results Master

This file is the numerical source of truth for the frozen Compact ORCA paper. Do not replace these values with older 39-sequence, 56-sequence, mean/std/max/delta, or pre-fix results.

## Frozen Protocol

- Dataset: 571 sequences, 26,260 frames, six Chinese dance gesture classes.
- Development/final split: 456/115 sequences.
- Primary encoder: Resample16.
- Primary setting: 3 shots per class, 20 repeated common splits.
- Classifiers: SVM, KNN, RandomForest, MLP.
- Compact selection: development data only.
- Frozen Compact Refined-7 indices: `[3, 6, 9, 11, 12, 15, 16]`.
- Encoded sizes: Refined ORCA-17 = 272, JointAngle-11 = 176, Compact Refined-7 = 112.

## Temporal Stability

| Actuator representation | Velocity mean | Acceleration mean |
|---|---:|---:|
| Actuator Projection-17 | 0.459021 | 0.709996 |
| Refined ORCA-17 | 0.246964 | 0.270325 |

- Velocity reduction: 46.2%.
- Acceleration reduction: 61.9%.
- These values are comparable because both representations use the same 17-dimensional actuator space.

## Controlled Perturbation Sensitivity

| Corruption | Actuator Projection-17 | Refined ORCA-17 | Reduction |
|---|---:|---:|---:|
| Overall | 0.0292 | 0.0182 | 37.6% |
| Gaussian | 0.1020 | 0.0605 | 40.7% |
| Spike | 0.0060 | 0.0050 | 17.7% |
| Dropout | 0.0059 | 0.0046 | 21.9% |

Interpret as actuator sensitivity to controlled landmark corruption, not physical ground-truth recovery.

## Frozen Final Classification

Values are Accuracy / Macro-F1 / Kappa.

| Representation | SVM | KNN | RandomForest | MLP |
|---|---|---|---|---|
| JointAngle-11 | .8026 / .8082 / .7629 | .7622 / .7605 / .7142 | .8322 / .8331 / .7985 | .7961 / .7972 / .7549 |
| Actuator Projection-17 | .7883 / .7946 / .7460 | .7552 / .7453 / .7063 | .8330 / .8317 / .7996 | .7961 / .7939 / .7553 |
| Refined ORCA-17 | .8013 / .8068 / .7615 | .7457 / .7336 / .6947 | .8252 / .8249 / .7902 | .7748 / .7721 / .7298 |
| Projection Flex-11 | .8096 / .8154 / .7714 | .7800 / .7805 / .7357 | .8348 / .8359 / .8017 | .8122 / .8106 / .7744 |
| Refined Flex-11 | .8117 / .8177 / .7740 | .7765 / .7766 / .7315 | .8209 / .8215 / .7850 | .8165 / .8168 / .7796 |
| Compact Projection-7 | .8339 / .8382 / .8004 | .8039 / .8066 / .7642 | .8339 / .8335 / .8006 | .8061 / .8026 / .7668 |
| Compact Refined-7 | .8400 / .8438 / .8077 | .8070 / .8091 / .7678 | .8235 / .8222 / .7881 | .8143 / .8107 / .7768 |

The exact machine-readable values, including standard deviations and confidence intervals, are in `tables/table_04_final_classification.csv`.

## Compact Refined-7 Versus JointAngle-11

Accuracy differences are Compact Refined-7 minus JointAngle-11.

| Classifier | Difference | 95% CI | Holm p | Cohen dz | Conclusion |
|---|---:|---:|---:|---:|---|
| SVM | +0.0374 | [+0.0182, +0.0566] | 0.0111 | +0.854 | Significant |
| KNN | +0.0448 | [+0.0277, +0.0619] | 0.0021 | +1.151 | Significant |
| RandomForest | -0.0087 | [-0.0282, +0.0108] | 1.0000 | -0.195 | Not significant |
| MLP | +0.0183 | [+0.0014, +0.0352] | 0.1397 | +0.475 | Not significant after Holm |

Macro-F1 follows the same significance pattern: significant for SVM and KNN, not significant for RandomForest and MLP.

## Runtime

- Frames: 300.
- Mean: 27.607 ms/frame.
- Median: 27.269 ms/frame.
- 95th percentile: 32.620 ms/frame.
- Mean iterations: 6.02.
- Optimizer success rate: 100%.
- Finite-output rate: 100%.

Use `causal frame-wise refinement`; do not make a hardware-independent real-time claim.

## Source Files

- Frozen classification: `diagnostics/orca_compact_selection_20260827/final_test_results.csv`
- Paired statistics: `diagnostics/orca_compact_selection_20260827/final_test_paired_comparisons_holm.csv`
- Development selection: `diagnostics/orca_compact_selection_20260827/compact_dimension_selection.csv`
- Stability: `diagnostics/updated_6class_20260820/UPDATED_RESULTS.md`
- Perturbation: `diagnostics/recovery_benchmark_20260827/results/actuator_overall_summary.csv`
- Runtime: `diagnostics/updated_6class_20260820/paper_completion/runtime_summary_6class.json`
