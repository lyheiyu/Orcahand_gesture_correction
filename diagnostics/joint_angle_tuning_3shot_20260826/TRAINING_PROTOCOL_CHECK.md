# Joint-angle and classifier training protocol check

## Scope

This diagnostic asks whether the 3-shot JointAngle-11 result is caused by an
incorrect or unfair classifier training protocol. It does not modify the ORCA
optimizer or any loss weight.

## Relationship to the reference method

The implemented JointAngle-11 feature uses the requested absolute three-dimensional
included angle:

```text
v1 = proximal - joint
v2 = distal - joint
theta = degrees(acos(clip(dot(v1, v2) / (norm(v1) * norm(v2)), -1, 1)))
```

It contains thumb CMC/MCP/IP and MCP/PIP for the other four fingers. DIP angles
are excluded. This matches the angle definition and topology requested from the
reference as closely as the MediaPipe topology permits.

The experiment is not a reproduction of a classification experiment from the
reference paper. The reference is a hand-kinematics data descriptor. In this
project, the angles are intentionally evaluated without additional smoothing,
using the same sequence splits, temporal pooling, scaling, and classifiers as
the ORCA representations. Therefore, it is an adapted conventional geometric
baseline. It has not been strengthened with ORCA, MuJoCo, or label information.

## Fixed protocol

The original comparison used fixed classifier parameters for every
representation:

- SVM: RBF, C=5, gamma=scale;
- KNN: k=3, distance weighting;
- MLP: hidden sizes 128/64, alpha=1e-4.

StandardScaler is fitted only on the external training set. PCA is fitted only
on training frames. The same sequence-level train/test split is reused across
representations.

## Nested tuning diagnostic

To test whether fixed hyperparameters systematically disadvantage Optimized
Action, all representations were evaluated with the same train-only nested
search. For each of 20 external 3-shot splits, a three-fold stratified search
was performed inside the 18-sequence training set. Macro-F1 selected the
parameters. The external test set was not used for parameter selection.

### Accuracy

| Classifier | Representation | Fixed | Nested tuned | Change |
|---|---|---:|---:|---:|
| SVM | JointAngle-11 | 0.7239 | 0.7083 | -0.0157 |
| SVM | Corrected-17 | 0.7500 | 0.6874 | -0.0626 |
| SVM | Optimized Action-17 | 0.7252 | 0.6622 | -0.0630 |
| KNN | JointAngle-11 | 0.6883 | 0.7035 | +0.0152 |
| KNN | Corrected-17 | 0.6857 | 0.6913 | +0.0057 |
| KNN | Optimized Action-17 | 0.6787 | 0.6974 | +0.0187 |
| MLP | JointAngle-11 | 0.7296 | 0.7309 | +0.0013 |
| MLP | Corrected-17 | 0.7326 | 0.7209 | -0.0117 |
| MLP | Optimized Action-17 | 0.7248 | 0.6909 | -0.0339 |

Nested tuning did not make Optimized Action consistently outperform
JointAngle-11. KNN improved, but JointAngle-11 improved at the same time. SVM
and MLP tuning was unstable because only three training examples per class are
available to the inner validation procedure.

## Interpretation

The fixed protocol is not confirmed as the cause of the JointAngle result. The
main limitation is representation-level: existing diagnostics show that the
optimized actuator trajectory retains only about 49--53% of the mean motion
amplitude of Corrected on the inspected sequences. The optimizer therefore
trades discriminative motion variation for stability.

The loss diagnostics also show that the default-pose and heuristic-prior terms
are the largest auxiliary weighted contributions. A future development study
may search these weights, together with temporal weights, on a development set.
That search must be confirmed on new or untouched test data; tuning the weights
against the repeatedly inspected current test results would not constitute a
valid improvement.

## Reporting recommendation

- Keep RandomForest in the supplementary classifier suite rather than deleting
  it after observing the result.
- Use SVM as the pre-declared primary classifier if the paper is centered on
  high-dimensional few-shot classification.
- Retain JointAngle-11 as a strong conventional baseline.
- Describe Optimized Action as a stability--discrimination trade-off under the
  current fixed-weight optimizer.
- If higher clean-data classification is required as the primary claim, collect
  an untouched validation/test partition before tuning optimizer weights.
