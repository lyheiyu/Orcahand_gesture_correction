# Final Claims and Limitations

## Safe Central Claim

The ORCA actuator projection provides a compact semantic representation of MediaPipe landmarks; MuJoCo-constrained causal refinement reduces actuator-space temporal variation and sensitivity to controlled landmark corruption; and a seven-actuator subset frozen on development data improves final-test performance over JointAngle-11 for SVM and KNN, while RandomForest and MLP differences are not significant after Holm correction.

## Claims Supported by Current Evidence

- The 17-dimensional actuator projection is a strong structured representation under the current acquisition protocol.
- Refined ORCA-17 is smoother than Actuator Projection-17 within the same actuator space.
- Optimized Action is less sensitive to the tested Gaussian, spike, and dropout corruptions.
- Compact Refined-7 uses 112 encoded features, 58.8% fewer than Refined ORCA-17 and 36.4% fewer than JointAngle-11.
- Compact Refined-7 significantly outperforms JointAngle-11 for SVM and KNN in the frozen test.
- RandomForest and MLP do not show a significant Compact Refined-7 versus JointAngle-11 difference after Holm correction.
- The optimizer completed the diagnostic runtime sample without NaN/Inf and respected hard bounds.

## Claims Not Supported

- Exact or ground-truth 3D human hand pose recovery.
- Universal superiority over joint-angle, PCA, or smoothing methods.
- Subject-independent generalization.
- Full dynamics-based physical correction involving force, torque, contact, or inertia.
- General left/right hand canonicalization.
- Real-world occlusion completion based on landmark confidence.
- A full reproduction of the external JointAngle paper.
- Hardware-independent real-time operation.

## Required Terminology

- Prefer `kinematically constrained`, `model-consistent`, or `MuJoCo-constrained` for the implemented objective.
- `Physics-informed` is acceptable only when immediately qualified as forward-kinematic and actuator-constrained.
- Use `Actuator Projection-17` for the code feature set `corrected`.
- Use `Optimized Action` for the refined 17-dimensional actuator state.
- Use `Compact Refined-7` for the frozen seven-channel refined readout.
- Use `controlled perturbation sensitivity`, not `physical recovery`.
- Use `causal frame-wise temporal refinement`, not independent single-frame correction.

## Top Limitations

1. Participant and session diversity is insufficient for subject-independent claims.
2. No synchronized ground-truth 3D human pose is available.
3. Robotic and human hand geometries do not match exactly.
4. Compact7 requires external validation.
5. The right-hand coordinate convention is acquisition specific.
6. JointAngle-11 is a definition-level baseline, not a complete system reproduction.
7. Resample16 captures coarse order but is not a learned temporal encoder.
8. Synthetic corruptions do not cover all real occlusions.
