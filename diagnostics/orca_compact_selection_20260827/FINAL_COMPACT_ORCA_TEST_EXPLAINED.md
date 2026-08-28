# Final Compact ORCA Test Explained

## A. What was frozen before final testing?

K*=7, actuator indices `[3, 6, 9, 11, 12, 15, 16]`, the 456/115 outer split, Resample-16, 3-shot sampling, 20 seeds, classifier settings, utility formula, and semantic selection rule were frozen. Manifest hashes were verified before execution.

## B. Why could the final test not influence selection?

Actuator utility and K were derived exclusively from nested development validation. The final program loaded the frozen subset directly and contains no ranking or K-selection call. It also refuses to run if final output files already exist.

## C. Why was K=7 selected?

K=7 had the highest four-classifier development combined score (0.8088); K=9 was nearly identical at 0.8085. The predefined smallest-within-one-standard-error rule therefore froze K=7.

## D. Which actuators were selected?

The seven coordinates are little PIP (3), ring PIP (6), middle PIP (9), index MCP/PIP (11/12), and thumb MCP/IP (15/16). They describe finger flexion and retain coverage of all five fingers.

## E. How is 7D different from JointAngle-11?

Compact OA-7 contains seven ORCA actuator coordinates after MuJoCo-constrained refinement. JointAngle-11 contains eleven angles computed directly from triples of MediaPipe landmarks. They differ in dimension, semantics, and whether temporal refinement is applied.

## F. Why is the comparison fair?

All representations use identical 3-shot training sequence IDs in every repeat, the same frozen 115-sequence test set, Resample-16, training-only scaling, and fixed classifier settings. No classifier is tuned for Compact ORCA.

## G. Exact final-test protocol

Each of 20 repeats samples three development sequences per class using seed `42 + repeat`. The frozen final test never changes. Models are trained separately for SVM, KNN, RandomForest, and MLP. Repeat, not frame, is the statistical unit.

## H. Final results for all representations

Values are mean Accuracy / Macro-F1 / Kappa.

| Representation | SVM | KNN | RandomForest | MLP |
|---|---|---|---|---|
| JointAngle-11 | 0.8026 / 0.8082 / 0.7629 | 0.7622 / 0.7605 / 0.7142 | 0.8322 / 0.8331 / 0.7985 | 0.7961 / 0.7972 / 0.7549 |
| Corrected-17 | 0.7883 / 0.7946 / 0.7460 | 0.7552 / 0.7453 / 0.7063 | 0.8330 / 0.8317 / 0.7996 | 0.7961 / 0.7939 / 0.7553 |
| OptimizedAction-17 | 0.8013 / 0.8068 / 0.7615 | 0.7457 / 0.7336 / 0.6947 | 0.8252 / 0.8249 / 0.7902 | 0.7748 / 0.7721 / 0.7298 |
| Corrected-Flex11 | 0.8096 / 0.8154 / 0.7714 | 0.7800 / 0.7805 / 0.7357 | 0.8348 / 0.8359 / 0.8017 | 0.8122 / 0.8106 / 0.7744 |
| OptimizedAction-Flex11 | 0.8117 / 0.8177 / 0.7740 | 0.7765 / 0.7766 / 0.7315 | 0.8209 / 0.8215 / 0.7850 | 0.8165 / 0.8168 / 0.7796 |
| Compact Corrected-7 | 0.8339 / 0.8382 / 0.8004 | 0.8039 / 0.8066 / 0.7642 | 0.8339 / 0.8335 / 0.8006 | 0.8061 / 0.8026 / 0.7668 |
| Compact OA-7 | 0.8400 / 0.8438 / 0.8077 | 0.8070 / 0.8091 / 0.7678 | 0.8235 / 0.8222 / 0.7881 | 0.8143 / 0.8107 / 0.7768 |
| Corrected-PCA11 | 0.7687 / 0.7769 / 0.7224 | 0.7387 / 0.7326 / 0.6858 | 0.8139 / 0.8159 / 0.7765 | 0.7691 / 0.7695 / 0.7225 |
| OptimizedAction-PCA11 | 0.7826 / 0.7901 / 0.7390 | 0.7409 / 0.7422 / 0.6885 | 0.7952 / 0.7970 / 0.7541 | 0.7613 / 0.7629 / 0.7132 |

## I-L. Primary paired comparisons

Differences are first representation minus second representation.

| Classifier | Metric | Comparison | Difference | 95% CI | Raw p | Holm p | dz |
|---|---|---|---:|---:|---:|---:|---:|
| svm | accuracy | compact_optimized_action7_minus_joint_angle11 | +0.0374 | 0.0192 | 0.002211 | 0.01106 | +0.854 |
| svm | macro_f1 | compact_optimized_action7_minus_joint_angle11 | +0.0356 | 0.0185 | 0.0008507 | 0.004253 | +0.844 |
| svm | accuracy | compact_optimized_action7_minus_optimized_action_flex11 | +0.0283 | 0.0164 | 0.004089 | 0.01315 | +0.757 |
| svm | macro_f1 | compact_optimized_action7_minus_optimized_action_flex11 | +0.0262 | 0.0157 | 0.003153 | 0.01261 | +0.729 |
| svm | accuracy | compact_optimized_action7_minus_optimized_action17 | +0.0387 | 0.0202 | 0.003288 | 0.01315 | +0.840 |
| svm | macro_f1 | compact_optimized_action7_minus_optimized_action17 | +0.0371 | 0.0204 | 0.003654 | 0.01261 | +0.797 |
| svm | accuracy | compact_corrected7_minus_compact_optimized_action7 | -0.0061 | 0.0061 | 0.0335 | 0.0335 | -0.439 |
| svm | macro_f1 | compact_corrected7_minus_compact_optimized_action7 | -0.0056 | 0.0057 | 0.1054 | 0.1054 | -0.433 |
| knn | accuracy | compact_optimized_action7_minus_joint_angle11 | +0.0448 | 0.0171 | 0.000425 | 0.002125 | +1.151 |
| knn | macro_f1 | compact_optimized_action7_minus_joint_angle11 | +0.0486 | 0.0205 | 0.0001678 | 0.0006714 | +1.038 |
| knn | accuracy | compact_optimized_action7_minus_optimized_action_flex11 | +0.0304 | 0.0198 | 0.008308 | 0.02493 | +0.675 |
| knn | macro_f1 | compact_optimized_action7_minus_optimized_action_flex11 | +0.0325 | 0.0215 | 0.008308 | 0.02493 | +0.663 |
| knn | accuracy | compact_optimized_action7_minus_optimized_action17 | +0.0613 | 0.0210 | 0.0002121 | 0.001273 | +1.279 |
| knn | macro_f1 | compact_optimized_action7_minus_optimized_action17 | +0.0755 | 0.0248 | 1.907e-05 | 0.0001144 | +1.333 |
| knn | accuracy | compact_corrected7_minus_compact_optimized_action7 | -0.0030 | 0.0083 | 0.4601 | 0.4601 | -0.160 |
| knn | macro_f1 | compact_corrected7_minus_compact_optimized_action7 | -0.0025 | 0.0085 | 0.6477 | 0.6477 | -0.128 |
| rf | accuracy | compact_optimized_action7_minus_joint_angle11 | -0.0087 | 0.0195 | 0.3753 | 1 | -0.195 |
| rf | macro_f1 | compact_optimized_action7_minus_joint_angle11 | -0.0109 | 0.0204 | 0.2611 | 1 | -0.234 |
| rf | accuracy | compact_optimized_action7_minus_optimized_action_flex11 | +0.0026 | 0.0175 | 0.9839 | 1 | +0.065 |
| rf | macro_f1 | compact_optimized_action7_minus_optimized_action_flex11 | +0.0008 | 0.0183 | 0.8695 | 1 | +0.018 |
| rf | accuracy | compact_optimized_action7_minus_optimized_action17 | -0.0017 | 0.0197 | 0.8877 | 1 | -0.039 |
| rf | macro_f1 | compact_optimized_action7_minus_optimized_action17 | -0.0027 | 0.0201 | 0.8983 | 1 | -0.059 |
| rf | accuracy | compact_corrected7_minus_compact_optimized_action7 | +0.0104 | 0.0105 | 0.06718 | 0.4031 | +0.434 |
| rf | macro_f1 | compact_corrected7_minus_compact_optimized_action7 | +0.0113 | 0.0108 | 0.05341 | 0.3204 | +0.458 |
| mlp | accuracy | compact_optimized_action7_minus_joint_angle11 | +0.0183 | 0.0169 | 0.02794 | 0.1397 | +0.475 |
| mlp | macro_f1 | compact_optimized_action7_minus_joint_angle11 | +0.0135 | 0.0171 | 0.1893 | 0.7574 | +0.345 |
| mlp | accuracy | compact_optimized_action7_minus_optimized_action_flex11 | -0.0022 | 0.0217 | 0.8103 | 1 | -0.044 |
| mlp | macro_f1 | compact_optimized_action7_minus_optimized_action_flex11 | -0.0061 | 0.0215 | 0.498 | 1 | -0.125 |
| mlp | accuracy | compact_optimized_action7_minus_optimized_action17 | +0.0396 | 0.0203 | 0.004212 | 0.02527 | +0.856 |
| mlp | macro_f1 | compact_optimized_action7_minus_optimized_action17 | +0.0386 | 0.0216 | 0.004221 | 0.02533 | +0.783 |
| mlp | accuracy | compact_corrected7_minus_compact_optimized_action7 | -0.0083 | 0.0069 | 0.03911 | 0.1564 | -0.522 |
| mlp | macro_f1 | compact_corrected7_minus_compact_optimized_action7 | -0.0080 | 0.0073 | 0.03999 | 0.1999 | -0.484 |

## M. Statistical interpretation

Wilcoxon tests are paired by repeat. Holm adjustment is applied within each classifier-and-metric family across the six predefined primary comparisons. Raw p-values are not treated as sufficient when Holm-adjusted p-values are non-significant.

## N. Overall outcome

The predefined decision rule classifies this result as **comparability**.

## O. Safe paper claim

> The frozen 7D compact refined ORCA representation achieved comparable recognition performance to JointAngle-11 while using 36.4% fewer frame-level dimensions.

## P. Claims that must not be made

Do not claim universal superiority, optimality over all actuator subsets, generalization beyond this six-class dataset, or that removed actuators are biologically unimportant. Do not alter K or the seven coordinates after these results.

## Q. Code locations

- `run_compact_orca_selection.py::_verify_frozen_gate`: validates hashes/spec and prevents reruns.
- `run_compact_orca_selection.py::evaluate_frozen_final`: applies the frozen subset and common repeats.
- `evaluate_sequence_encodings.py::encode_sequence`: performs Resample-16.
- `train_svm.py::_build_model`: fixed classifiers with training-only StandardScaler.
- `run_compact_orca_selection.py::_paired_final`: repeat-level Wilcoxon and effect size.
- `run_compact_orca_selection.py::_holm_adjust`: Holm correction within predefined families.

Core frozen selection code:

```python
FROZEN_INDICES = (3, 6, 9, 11, 12, 15, 16)
compact_sequence = actuator_sequence[:, FROZEN_INDICES]
encoded = resample16(compact_sequence)
```

## R. Plain-language conclusion

The frozen 7D compact refined ORCA representation achieved comparable recognition performance to JointAngle-11 while using 36.4% fewer frame-level dimensions.
