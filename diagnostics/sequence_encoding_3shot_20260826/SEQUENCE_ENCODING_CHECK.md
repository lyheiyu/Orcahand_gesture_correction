# Sequence encoding protocol check

## Question

The original lightweight protocol maps a sequence to global mean, standard
deviation, maximum, and end-minus-start statistics. This is a valid order-light
baseline, but it is not sufficient as the only downstream protocol for a paper
about temporal refinement.

Global extrema may occur at different frames and therefore need not describe a
real hand state. Mean and standard deviation discard ordering, while the endpoint
difference cannot distinguish trajectories with identical endpoints. Adding a
minimum does not solve these limitations.

## Compared encodings

- `global4`: mean, standard deviation, maximum, endpoint difference;
- `global5`: global4 plus minimum;
- `pyramid`: mean and standard deviation in 1, 2, and 4 normalized temporal bins,
  plus endpoint difference;
- `resample16`: linear interpolation to 16 normalized time steps followed by
  vectorization.

All encodings use the same 20 external 3-shot sequence splits and the same fixed
SVM, KNN, RandomForest, and MLP settings. The encoding is applied identically to
JointAngle, Corrected, and Optimized Action. PCA remains train-only.

## Main result

| Classifier | Representation | Global statistics | Resampled trajectory | Change |
|---|---|---:|---:|---:|
| SVM | JointAngle-11 | 0.7239 | 0.7978 | +0.0739 |
| SVM | Corrected-17 | 0.7500 | 0.7752 | +0.0252 |
| SVM | Optimized Action-17 | 0.7252 | 0.7757 | +0.0504 |
| KNN | JointAngle-11 | 0.6883 | 0.7652 | +0.0770 |
| KNN | Corrected-17 | 0.6857 | 0.7448 | +0.0591 |
| KNN | Optimized Action-17 | 0.6787 | 0.7226 | +0.0439 |
| RandomForest | JointAngle-11 | 0.7930 | 0.8435 | +0.0504 |
| RandomForest | Corrected-17 | 0.7748 | 0.8348 | +0.0600 |
| RandomForest | Optimized Action-17 | 0.7683 | 0.8209 | +0.0526 |
| MLP | JointAngle-11 | 0.7296 | 0.7891 | +0.0596 |
| MLP | Corrected-17 | 0.7326 | 0.7952 | +0.0626 |
| MLP | Optimized Action-17 | 0.7248 | 0.7626 | +0.0378 |

The resampled trajectory significantly improves Optimized Action under all four
classifiers (`p < 0.012`). Under SVM, resampled Optimized Action and Corrected are
effectively identical (difference `+0.0004`, `p = 0.856`). JointAngle remains
`0.0222` higher than Optimized Action under SVM (`p = 0.0073`).

## Decision

The global statistics should remain as a lightweight, order-light baseline, not
as the sole primary sequence evaluation. A normalized resampled trajectory is a
better conventional primary protocol because it preserves coarse temporal order
without training a separate temporal neural network.

This exploratory comparison has been inspected on the current test data. The
choice of 16 time steps should therefore be frozen and confirmed on new or
untouched data before it is used for a final confirmatory paper claim. It would
not be valid to scan many target lengths on the same test set and report only the
best one.

The result also confirms that sequence encoding was only part of the previous
gap. It improves every representation and does not make Optimized Action
universally dominant. The remaining interpretation is still a
stability--discrimination trade-off.
