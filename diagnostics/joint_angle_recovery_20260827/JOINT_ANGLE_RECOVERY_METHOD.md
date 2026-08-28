# Joint-Angle Recovery Smoke Diagnostic

## Scope

This experiment evaluates recovery toward an uncorrupted JointAngle-11 trajectory used as a controlled clean reference. It is not anatomical or physical ground truth. No classification experiment was run.

Six frozen-test sequences were selected, one from each gesture class. The three requested corruption conditions were assigned twice each. All thresholds, empirical bounds, confidence strength, Kalman parameters, and One-Euro parameters were selected using development sequences only.

## Implemented Method

Each frame is converted directly from the 21 MediaPipe landmarks to the existing 11 absolute 3D joint angles. For a joint formed by points `(a,b,c)`, the angle is

```text
theta = acos(clip(dot(a-b, c-b) / (||a-b|| ||c-b||), -1, 1)).
```

Angles are represented in degrees. No smoothing is applied during extraction. Zero-length vectors are marked invalid and handled without NaN/Inf.

For frame `t >= 2`, the causal prediction is

```text
theta_pred[t] = 2 theta_hat[t-1] - theta_hat[t-2].
```

The second frame uses the previous refined state. For each joint, automatic confidence is

```text
w[t] = exp(-s * (
    0.35 |theta_obs[t] - theta_hat[t-1]| / jump_scale
  + 0.30 geometry_error[t] / geometry_scale
  + 0.35 |theta_obs[t] - theta_pred[t]| / prediction_scale
)).
```

`geometry_error` is the maximum absolute log-change in the two bone-vector lengths defining the angle. Confidence is clipped to `[0,1]`; invalid geometry receives zero confidence. The refined state is

```text
theta_hat[t] = w[t] theta_obs[t] + (1-w[t]) theta_pred[t],
```

followed by empirical feasibility clipping. The bounds are the development-set 0.5th and 99.5th angle percentiles with a 5-degree margin, clipped to `[0,180]`. They are empirical data-supported bounds, not anatomical limits.

The Oracle uses the same update but sets `w=0` when any landmark defining that joint is listed in the synthetic corruption manifest and `w=1` otherwise. Only the Oracle reads the mask.

## Development and Frozen Test Protocol

- Development sequences: `457`
- Frozen test sequences: `114`
- Smoke sequences: `6`, selected from the frozen test set with one sequence per gesture class
- Development calibration subset: `18`, three per class
- Confidence strength candidates: `0.25, 0.5, 1.0, 2.0`; selected value: `2.0`
- Kalman development grid: process variance `0.01, 0.1, 1, 10`; measurement variance `0.1, 1, 10, 100`; selected values: `1` and `100`
- One-Euro development grid: minimum cutoff `0.5, 1, 2, 4`; beta `0, 0.02, 0.1, 0.5`; selected values: `0.5` and `0`

The final smoke conditions are medium Gaussian landmark noise (`sigma=0.03`), a 3-frame single-landmark spike (`magnitude=0.75`), and 5-frame frozen-finger occlusion. Each condition is evaluated on two frozen-test sequences. Exact affected landmarks are stored in `corruption_manifest.csv`.

## First Diagnostic Result

| Method | Corrupted-joint MAE | Recovery ratio | Velocity error | Median amplitude retention | Lag (frames) |
|---|---:|---:|---:|---:|---:|
| Corrupted | 22.555 | 0.000 | 3.875 | 1.202 | 0.00 |
| Kalman | 9.941 | -0.084 | 3.601 | 0.423 | 3.67 |
| One-Euro | 12.873 | 0.150 | 3.478 | 0.670 | 1.50 |
| Automatic Confidence | 7.543 | 0.256 | 3.878 | 1.017 | 1.83 |
| Oracle Confidence | 11.623 | -0.469 | 1.051 | 0.667 | 0.00 |

The overall average should not be read without the corruption-specific results:

| Method | Gaussian recovery ratio | Spike recovery ratio | 5-frame occlusion recovery ratio |
|---|---:|---:|---:|
| Kalman | -0.024 | 0.726 | -0.954 |
| One-Euro | 0.102 | 0.533 | -0.185 |
| Automatic Confidence | 0.006 | 0.756 | 0.005 |
| Oracle Confidence | -0.467 | 0.676 | -1.616 |

Automatic Confidence is promising for isolated spikes: its mean recovery ratio is `0.756`, its median amplitude retention is `0.989`, and one spike sequence reaches a recovery ratio of `0.958`. It does not provide meaningful recovery for Gaussian noise or frozen-finger occlusion in this smoke test.

## Decision

Outcome C: Oracle does not outperform the strongest conventional filter; the constant-velocity temporal model is insufficient for at least part of this diagnostic.

Results must also be inspected by corruption type because an oracle that ignores every Gaussian-corrupted observation can fail differently from an oracle handling a short spike or occlusion.

## Failure Analysis

The result identifies two separate limitations.

First, frozen-finger occlusion remains locally geometry-consistent. The finger bone lengths and angles stop changing together, so the current angle-jump, bone-length-change, and prediction-disagreement signals often regard the stale observation as reliable. Mean automatic confidence on the affected joints is approximately `0.98` in the two occlusion cases. Detecting this failure requires an additional cue, such as MediaPipe visibility/presence, image-space tracking confidence, cross-finger motion context, or an explicit stale-observation detector.

Second, the Oracle result shows that confidence alone is insufficient. Setting confidence to zero for five frames forces pure constant-velocity extrapolation. This extrapolation can drift when the real finger decelerates or reverses direction, causing negative occlusion recovery ratios. A damped-velocity, hold/decay, Kalman state model, or short-window smoother should be tested before any full benchmark.

For Gaussian corruption, the oracle marks all affected observations as completely unreliable and therefore ignores the whole noisy trajectory. This diagnostic is intentionally strict, but it explains why Oracle Confidence collapses motion amplitude for Gaussian cases. A reliability value between zero and one would be more appropriate for noisy-but-present observations; that would be a different oracle definition and should be specified before another experiment.

Automatic Confidence also modifies visible joints: its visible-joint MAE is non-zero because normal motion can lower confidence. This needs to be reduced before claiming preservation of reliable observations.

## Milestone Decision

Do not run the full 571-sequence benchmark with this exact formulation yet. The weighted fusion itself is useful for short isolated spikes, but the current confidence estimator cannot detect stale occlusion and the constant-velocity predictor cannot safely bridge five-frame missing intervals. Per the predefined decision rule, the next experiment should modify only the predictor/confidence diagnosis on the development set and repeat this same frozen smoke test.

## Validation

- Non-finite values: `0`
- Confidence range violations: `0`
- Causal future-perturbation difference: `0.000e+00`
- Sequence state is reset because every recovery call creates local state.
- The automatic method never reads the synthetic corruption mask; only Oracle Confidence receives it.
