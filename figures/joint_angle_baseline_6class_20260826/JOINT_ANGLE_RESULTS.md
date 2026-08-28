# JointAngle-11 Baseline Results

Primary comparison: 3-shot, with the same nested sequence splits and preprocessing rules.

Invalid angle values: 0 across 26260 frames.

## Primary Accuracy

| Classifier | Raw | PCA-11 | JointAngle-11 | PCA-17 | Corrected | Optimized Action | Optimized Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM | 0.3967 | 0.5868 | 0.7254 | 0.6163 | 0.7405 | 0.7179 | 0.4447 |
| KNN | 0.3590 | 0.5452 | 0.6791 | 0.5814 | 0.6866 | 0.6710 | 0.4118 |
| RandomForest | 0.4228 | 0.6190 | 0.7856 | 0.6593 | 0.7621 | 0.7527 | 0.4817 |
| MLP | 0.3910 | 0.5897 | 0.7287 | 0.6266 | 0.7311 | 0.7033 | 0.4577 |

## Paired Accuracy Differences

Positive values favor the first representation. Differences are percentage points.

| Classifier | Comparison | Difference (pp) | 95% CI (pp) | Wilcoxon p |
|---|---|---:|---:|---:|
| SVM | joint angle minus raw | 32.87 | 2.12 | 7.53e-10 |
| SVM | joint angle minus raw pca11 | 13.86 | 1.88 | 1.111e-09 |
| SVM | corrected minus joint angle | 1.51 | 1.34 | 0.02534 |
| SVM | optimized action minus joint angle | -0.75 | 1.35 | 0.3418 |
| SVM | optimized action minus corrected | -2.26 | 0.94 | 3.134e-05 |
| KNN | joint angle minus raw | 32.02 | 1.97 | 7.538e-10 |
| KNN | joint angle minus raw pca11 | 13.39 | 1.96 | 1.41e-09 |
| KNN | corrected minus joint angle | 0.75 | 1.40 | 0.3042 |
| KNN | optimized action minus joint angle | -0.82 | 1.34 | 0.2789 |
| KNN | optimized action minus corrected | -1.57 | 1.01 | 0.007237 |
| RandomForest | joint angle minus raw | 36.28 | 2.00 | 7.54e-10 |
| RandomForest | joint angle minus raw pca11 | 16.66 | 2.12 | 7.474e-10 |
| RandomForest | corrected minus joint angle | -2.35 | 1.41 | 0.003417 |
| RandomForest | optimized action minus joint angle | -3.29 | 1.30 | 3.532e-05 |
| RandomForest | optimized action minus corrected | -0.94 | 0.95 | 0.07878 |
| MLP | joint angle minus raw | 33.77 | 1.96 | 7.521e-10 |
| MLP | joint angle minus raw pca11 | 13.90 | 1.80 | 7.496e-10 |
| MLP | corrected minus joint angle | 0.24 | 1.37 | 0.8476 |
| MLP | optimized action minus joint angle | -2.54 | 1.55 | 0.004313 |
| MLP | optimized action minus corrected | -2.78 | 0.98 | 1.02e-05 |

## Interpretation Rule

JointAngle-11 tests conventional geometric reparameterization without ORCA or MuJoCo. Corrected tests embodiment-aware actuator mapping, while Optimized Action additionally uses MuJoCo and causal temporal regularization. Absolute temporal-difference magnitudes in degrees must not be compared directly with actuator-space magnitudes.
