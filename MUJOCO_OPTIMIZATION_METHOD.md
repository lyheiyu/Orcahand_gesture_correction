# MuJoCo Optimization Method

## Motivation

The current `corrected` feature pipeline is a heuristic ORCA-aware projection:

1. extract geometric cues from MediaPipe landmarks
2. map them into the ORCA actuator space
3. clip them to the valid actuator ranges

This introduces embodiment constraints, but it is not yet a true optimization-based correction method.

To obtain a stronger and more defensible method for research, we upgrade the correction step to a MuJoCo-based pose fitting problem.


## State Variable

Let:

- `x_t` denote the MediaPipe landmark observation at frame `t`
- `q_t in R^17` denote the ORCA hand actuator-state vector at frame `t`

The 17-dimensional vector corresponds to the controllable ORCA joints:

- wrist
- pinky abd / mcp / pip
- ring abd / mcp / pip
- middle abd / mcp / pip
- index abd / mcp / pip
- thumb cmc / abd / mcp / pip


## Forward Model

Given a candidate state `q_t`, we construct the ORCA hand pose in MuJoCo and call:

- `mujoco.mj_forward(model, data)`

This yields the forward-kinematic positions of selected ORCA bodies. We use a sparse set of correspondences:

- wrist
- thumb tip
- index MCP
- index tip
- middle MCP
- middle tip
- pinky MCP
- pinky tip

Let `p_i(q_t)` denote the normalized 3D position of ORCA correspondence `i` under state `q_t`.

Let `y_i(x_t)` denote the normalized 3D position of the corresponding MediaPipe landmark.


## Objective Function

The optimized hand state is obtained by:

`q_t* = argmin_q L_t(q)`

with:

`L_t(q) = lambda_landmark * L_landmark(q, x_t)
        + lambda_palm * L_palm(q, x_t)
        + lambda_prior * L_prior(q, q_hat_t)
        + lambda_temporal * L_temporal(q, q_{t-1}*)`

where:

### Landmark alignment

`L_landmark(q, x_t) = sum_i || p_i(q) - y_i(x_t) ||_2^2`

This matches ORCA forward-kinematic keypoints to MediaPipe observations.


### Palm alignment

Let `n_orca(q)` be the normalized palm normal induced by the ORCA pose, and let `n_mp(x_t)` be the palm normal estimated from MediaPipe landmarks.

Then:

`L_palm(q, x_t) = || n_orca(q) - n_mp(x_t) ||_2^2`

This encourages correct palm orientation.


### Prior term

Let `q_hat_t` be the heuristic ORCA-aware projection obtained from the current rule-based corrected feature extractor.

Then:

`L_prior(q, q_hat_t) = || q - q_hat_t ||_2^2`

This stabilizes optimization and provides a good initialization.


### Temporal term

For sequential correction:

`L_temporal(q, q_{t-1}*) = || q - q_{t-1}* ||_2^2`

This penalizes abrupt changes and enforces temporal continuity.


## Constraints

The optimization is performed under MuJoCo/ORCA actuator limits:

`q_min <= q <= q_max`

where `q_min` and `q_max` are read directly from the ORCA actuator control ranges.

These bounds act as embodiment constraints and guarantee that optimized poses remain inside the feasible ORCA hand state space.


## Practical Interpretation

The method no longer directly maps landmarks to ORCA features. Instead, it searches for an ORCA pose that:

- matches observed landmarks
- matches palm orientation
- stays close to a structurally meaningful prior
- remains temporally smooth
- satisfies ORCA actuator bounds


## Relation to the Current Code

The code implementation is in:

- [mujoco_optimizer.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/src/orca_sim/mujoco_optimizer.py)

The current module supports:

- single-frame optimization
- sparse ORCA-to-MediaPipe landmark fitting
- palm-normal alignment
- prior and temporal penalties
- SciPy-based optimization when available
- coordinate-descent fallback otherwise


## Suggested Paper Wording

Safe wording:

- MuJoCo-based ORCA-constrained pose fitting
- optimization-based embodiment-constrained correction
- temporally regularized ORCA state projection

Avoid overstating as:

- exact physical reconstruction
- fully differentiable physics optimization

unless a stronger optimization and contact-aware formulation is implemented later.
