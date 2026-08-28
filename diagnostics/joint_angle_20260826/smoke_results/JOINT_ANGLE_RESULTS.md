# JointAngle-11 Baseline Results

Primary comparison: 3-shot, with the same nested sequence splits and preprocessing rules.

Invalid angle values: 0 across 26260 frames.

## Primary Accuracy

| Classifier | Raw | PCA-11 | JointAngle-11 | PCA-17 | Corrected | Optimized Action |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.3913 | 0.5478 | 0.7043 | 0.6130 | 0.7696 | 0.7391 |
| KNN | 0.3391 | 0.5217 | 0.7043 | 0.6087 | 0.7000 | 0.6739 |
| RandomForest | 0.4261 | 0.5913 | 0.7565 | 0.6826 | 0.8043 | 0.8087 |
| MLP | 0.4217 | 0.5522 | 0.7043 | 0.6565 | 0.7652 | 0.7478 |

## Interpretation Rule

JointAngle-11 tests conventional geometric reparameterization without ORCA or MuJoCo. Corrected tests embodiment-aware actuator mapping, while Optimized Action additionally uses MuJoCo and causal temporal regularization. Absolute temporal-difference magnitudes in degrees must not be compared directly with actuator-space magnitudes.
