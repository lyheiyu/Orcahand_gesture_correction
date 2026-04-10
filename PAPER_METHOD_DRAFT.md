# Draft Method Section

## Method

### 1. Problem Formulation

Let the observed MediaPipe hand landmarks at frame \(t\) be denoted by

\[
\mathbf{y}_t \in \mathbb{R}^{K \times 3},
\]

where \(K=21\) is the number of hand landmarks. These observations are convenient for real-time hand analysis, but they often exhibit frame-wise jitter, local geometric inconsistency, and unstable depth estimates under viewpoint change or partial occlusion.

Instead of directly using the observed landmarks as the final representation, we introduce a latent embodiment-constrained hand state

\[
\mathbf{q}_t \in \mathbb{R}^{d},
\]

where \(d=17\) corresponds to the actuator state of the ORCA robot hand. This latent state defines a lower-dimensional feasible hand manifold induced by the mechanical structure, actuator limits, and kinematic layout of the ORCA hand model.

Given a candidate state \(\mathbf{q}_t\), a MuJoCo forward model produces a reconstructed hand configuration:

\[
\hat{\mathbf{y}}_t = h(\mathbf{q}_t),
\]

where \(h(\cdot)\) is implemented through MuJoCo forward kinematics. Our goal is therefore not to smooth landmarks directly, but to estimate the most plausible ORCA hand state that explains the noisy observation and then re-project that state back to the landmark space:

\[
\mathbf{q}_t^* = \arg\min_{\mathbf{q}_t \in \mathcal{Q}} \mathcal{L}_t(\mathbf{q}_t), \qquad
\mathbf{y}_t^* = h(\mathbf{q}_t^*),
\]

where \(\mathcal{Q}\) is the feasible actuator space defined by ORCA control bounds.

This formulation allows the method to be interpreted as a structure-aware denoising process, in which noisy visual observations are projected onto an embodiment-constrained kinematic manifold.

### 2. Observation Model and Structural Prior

We model the MediaPipe landmarks as noisy observations of an underlying mechanically feasible hand configuration:

\[
\mathbf{y}_t = h(\mathbf{q}_t^{true}) + \boldsymbol{\epsilon}_t,
\]

where \(\boldsymbol{\epsilon}_t\) represents observation noise. In practice, this noise is not only additive image noise, but also includes depth ambiguity, landmark drift, and short-lived tracking failures. Directly learning or classifying from \(\mathbf{y}_t\) may therefore expose downstream models to non-physical landmark variations.

The ORCA hand model provides a useful structural prior for this problem. Because the robot hand has fixed kinematic topology, bounded joint motion, and limited actuation degrees of freedom, the set of realizable landmark configurations is much smaller than the unconstrained landmark space. Projecting observations into this state space suppresses geometrically implausible hand shapes and yields temporally more stable trajectories.

### 3. MuJoCo-Based Constrained State Estimation

For each frame, we estimate the latent hand state by minimizing the following objective:

\[
\mathcal{L}_t(\mathbf{q}) =
\lambda_l \mathcal{L}_{landmark}(\mathbf{q}, \mathbf{y}_t)
+ \lambda_n \mathcal{L}_{normal}(\mathbf{q}, \mathbf{y}_t)
+ \lambda_p \mathcal{L}_{prior}(\mathbf{q}, \tilde{\mathbf{q}}_t)
+ \lambda_s \mathcal{L}_{temporal}(\mathbf{q}, \mathbf{q}_{t-1}^*)
+ \lambda_d \mathcal{L}_{default}(\mathbf{q})
+ \lambda_b \mathcal{L}_{boundary}(\mathbf{q}),
\]

subject to

\[
\mathbf{q}^{min} \le \mathbf{q} \le \mathbf{q}^{max}.
\]

Here, \(\tilde{\mathbf{q}}_t\) is a heuristic initialization derived from geometric feature projection, while \(\mathbf{q}_{t-1}^*\) is the optimized state from the previous frame. The actuator bounds \(\mathbf{q}^{min}\) and \(\mathbf{q}^{max}\) are directly obtained from the ORCA MuJoCo model.

This formulation combines visual alignment, embodiment consistency, and temporal regularization within a single constrained optimization problem.

## Loss Design

### 1. Landmark Alignment Loss

The primary observation term encourages MuJoCo-reconstructed keypoints to match the observed MediaPipe landmarks:

\[
\mathcal{L}_{landmark}(\mathbf{q}, \mathbf{y}_t)
=
\sum_{i=1}^{K_s}
\left\| h_i(\mathbf{q}) - \mathbf{y}_{t,i} \right\|_2^2,
\]

where \(K_s\) is the number of sparse keypoint correspondences used in the current implementation. We use a sparse set consisting of the wrist, selected finger MCP points, and fingertip locations. This sparse fitting design is computationally efficient and already sufficient to constrain the major hand shape, while still allowing later extension to denser correspondences.

### 2. Palm Orientation Loss

Landmark alignment alone may not fully constrain the global orientation of the hand. To preserve palm orientation consistency, we define a palm-normal alignment term:

\[
\mathcal{L}_{normal}(\mathbf{q}, \mathbf{y}_t)
=
\left\|
\mathbf{n}_{orca}(\mathbf{q}) - \mathbf{n}_{mp}(\mathbf{y}_t)
\right\|_2^2,
\]

where \(\mathbf{n}_{orca}(\mathbf{q})\) is the palm normal computed from MuJoCo-reconstructed points and \(\mathbf{n}_{mp}(\mathbf{y}_t)\) is the palm normal estimated from MediaPipe landmarks. This term improves stability under palm flipping and out-of-plane rotation.

### 3. Geometric Prior Loss

To stabilize optimization and reduce ambiguity in the inverse problem, we use a heuristic ORCA-aware geometric projection as an initialization prior:

\[
\mathcal{L}_{prior}(\mathbf{q}, \tilde{\mathbf{q}}_t)
=
\left\|
\mathbf{q} - \tilde{\mathbf{q}}_t
\right\|_2^2.
\]

This prior is computed from hand-crafted geometric measurements extracted from MediaPipe landmarks, including finger flexion, finger spread, and a coarse wrist descriptor. It acts as a strong inductive bias that keeps the optimizer close to an embodiment-aware first guess.

### 4. Temporal Consistency Loss

Because hand tracking noise is often high-frequency and frame-local, temporal smoothness is essential:

\[
\mathcal{L}_{temporal}(\mathbf{q}, \mathbf{q}_{t-1}^*)
=
\left\|
\mathbf{q} - \mathbf{q}_{t-1}^*
\right\|_2^2.
\]

This term penalizes abrupt actuator changes and reduces frame-to-frame jitter in the reconstructed hand trajectory. In sequence settings, it effectively turns the optimizer into a regularized latent-state estimator rather than an isolated frame-wise fitter.

### 5. Default Pose Regularization

To prevent implausible drift toward extreme or unnecessary articulation, we additionally penalize deviation from the ORCA default pose:

\[
\mathcal{L}_{default}(\mathbf{q})
=
\left\|
\mathbf{q} - \mathbf{q}_{default}
\right\|_2^2.
\]

This term is particularly useful when the observation is weak or ambiguous, such as under partial occlusion or unstable wrist estimation.

### 6. Boundary Penalty

Although actuator limits are already enforced through box constraints, solutions may still accumulate near the limits. We therefore introduce an auxiliary boundary penalty:

\[
\mathcal{L}_{boundary}(\mathbf{q})
=
\sum_j \phi(q_j; q_j^{min}, q_j^{max}),
\]

where \(\phi(\cdot)\) softly penalizes states that approach the edges of the feasible control interval. This discourages saturation and improves naturalness of the optimized state.

### 7. Refined Landmark Reconstruction

After optimization, the final refined hand landmarks are obtained by forwarding the optimized state through the MuJoCo model:

\[
\mathbf{y}_t^* = h(\mathbf{q}_t^*).
\]

In the current implementation, we export both sparse optimized correspondences and a full 21-point reconstructed landmark set. Joint anchor positions are taken from MuJoCo kinematic anchors, and fingertip positions are read from the corresponding ORCA bodies. These reconstructed landmarks form the refined representation used for downstream analysis and classification experiments.

## Discussion

### 1. Why the Method Reduces Jitter

The proposed method reduces jitter for two complementary reasons. First, the temporal regularization explicitly suppresses frame-wise fluctuations in the latent hand state. Second, and more importantly, the embodiment prior rejects a large class of visually plausible but mechanically inconsistent landmark perturbations. As a result, the output is not merely smoothed in image space; it is projected into a physically interpretable and structurally feasible state space.

This is a key distinction from simple low-pass filtering. Direct landmark smoothing can reduce noise amplitude but cannot correct implausible finger geometry, inconsistent palm orientation, or unrealistic local distortions. Our method instead performs state estimation under structural constraints, which improves both stability and geometric coherence.

### 2. Scientific Positioning

The method should be described as a MuJoCo-based embodiment-constrained landmark refinement framework, rather than full physical reconstruction. In the current version, MuJoCo is used primarily as a forward kinematic and embodiment-consistency model. The optimization is performed in the ORCA actuator space and uses forward kinematics for evaluation, but it does not yet solve a full contact-aware dynamic reconstruction problem.

Therefore, the safest and most accurate claims are:

- the method performs structure-aware landmark denoising,
- it estimates a feasible latent hand state under ORCA constraints,
- it produces refined landmarks by re-projecting the optimized state through MuJoCo forward kinematics.

Claims such as exact hand pose recovery or fully physical reconstruction should be avoided unless stronger sensing or contact-aware estimation is introduced.

### 3. Limitations

Several limitations remain. First, the current correspondence design is still sparse, which may reduce observability for some local degrees of freedom. Second, the reconstructed 21-point landmarks are generated from the ORCA kinematic structure rather than from human anatomical ground truth, so they should be interpreted as structure-consistent refined landmarks rather than exact human landmarks. Third, the temporal model currently uses first-order regularization only; richer sequence models with velocity and acceleration penalties could further improve stability.

### 4. Expected Experimental Outcomes

Based on the method design, we expect three main effects in experiments. First, the refined landmarks should exhibit lower frame-wise velocity and acceleration energy than raw MediaPipe landmarks. Second, the refined representation should show reduced geometric inconsistency, such as more stable palm orientation and less local finger drift. Third, downstream gesture classification should benefit from the refined representation, especially in low-data or few-shot settings where noisy raw observations can disproportionately hurt learning.

Together, these properties support the use of the proposed framework as a research-oriented intermediate representation layer between visual hand tracking and downstream gesture recognition.
