# ORCA Dimension-Control Experiment Explained

## A. Why are we doing this experiment?

JointAngle has 11 frame-level dimensions, whereas the original ORCA representations have 17. After Resample-16, classifiers receive 176 versus 272 values. With only three training sequences per class, this dimensionality difference may matter. This experiment separates dimension, actuator semantics, and representation quality without changing the ORCA optimizer.

## B. What exactly is Corrected-17?

Corrected-17 is the frame-wise, rule-based mapping from normalized MediaPipe landmarks to the 17 right-hand ORCA actuator coordinates. It uses hand geometry and actuator ranges, but no temporal history and no MuJoCo optimization.

## C. What exactly is OptimizedAction-17?

OptimizedAction-17 starts from Corrected-17 and refines the same actuator state using MuJoCo forward kinematics, robust landmark fitting, priors, bounds, and causal temporal regularization. Its outputs remain 17 actuator values per frame.

## D. How is Flex11 created?

Flex11 is fixed before evaluation. It keeps thumb CMC/MCP/IP and MCP/PIP flexion for the four fingers. It removes wrist motion and six abduction-related coordinates. Both Corrected and Optimized Action use the exact same indices: `13,15,16,11,12,8,9,5,6,2,3`.

| Index | Actuator | Decision | Reason |
|---:|---|---|---|
| 0 | `right_wrist_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 1 | `right_p-abd_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 2 | `right_p-mcp_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 3 | `right_p-pip_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 4 | `right_r-abd_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 5 | `right_r-mcp_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 6 | `right_r-pip_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 7 | `right_m-abd_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 8 | `right_m-mcp_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 9 | `right_m-pip_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 10 | `right_i-abd_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 11 | `right_i-mcp_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 12 | `right_i-pip_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 13 | `right_t-cmc_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 14 | `right_t-abd_actuator` | no | wrist or abduction coordinate excluded by preregistered semantic rule |
| 15 | `right_t-mcp_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |
| 16 | `right_t-pip_actuator` | yes | flexion/opposition coordinate matched to JointAngle-11 |

## E. Why is Flex11 scientifically fair?

The subset is based on semantic correspondence with the preregistered JointAngle-11 definition, not on test accuracy. No alternative subsets are searched. It is therefore a controlled representation-selection test rather than test-set feature selection.

## F. How is PCA11 different?

Flex11 preserves 11 named actuator coordinates. PCA11 instead mixes all 17 coordinates into 11 orthogonal components. For every repeat, scaling and PCA are fitted only to training frames, then applied unchanged to test frames. PCA occurs before Resample-16.

## G. How does Resample-16 work?

Each variable-length trajectory is linearly interpolated at 16 normalized time positions and flattened in temporal order. Thus `17 x 16 = 272`, while `11 x 16 = 176`. Unlike global mean/std statistics, the order of the 16 temporal samples is retained.

## H. How is the 3-shot experiment performed?

For each of 2 repeats, the project creates one stratified sequence-level holdout, chooses exactly three training sequences per class from the training pool, fits preprocessing on those training sequences, trains SVM/KNN/RandomForest/MLP with fixed paper settings, and evaluates the common test sequences. All seven representations share the same split in every repeat.

## I. How should I interpret every possible result?

- ORCA-Flex11 above JointAngle-11: matched actuator semantics are at least as useful as direct human angles.
- Similar results: the original gap was substantially associated with dimension or irrelevant coordinates.
- JointAngle-11 above ORCA-Flex11: direct human-joint geometry better matches this recognition task.
- Flex11 above ORCA17: wrist/abduction dimensions add redundancy or noise in this few-shot setting.
- PCA11 above Flex11: variance-preserving mixtures are more useful than the predefined semantics.
- Flex11 above PCA11: named actuator semantics contribute beyond dimensionality reduction alone.

## J. Relevant code

- `generate_joint_angle_baseline.py::joint_angle_vector`: computes the 11 absolute 3D angles.
- `evaluate_orca_dimension_control.py::_load_base`: loads Corrected/OA and selects Flex11.
- `train_svm.py::_project_sequences_with_pca`: fits scaler and PCA on training frames only.
- `evaluate_sequence_encodings.py::encode_sequence`: Resample-16 and temporal flattening.
- `generate_shot_sweep_figures.py::_build_nested_splits`: shared sequence-level few-shot splits.
- `train_svm.py::_build_model`: training-only StandardScaler and fixed classifier.

## K. Final pipeline diagram

```text
MediaPipe landmarks -> JointAngle-11 ---------------------------> Resample16 -> classifier
                  \-> Corrected-17 -> Flex11/PCA11 ------------> Resample16 -> classifier
                                   \-> MuJoCo -> OA-17 -> Flex11/PCA11 -> classifier
```

## L. Results and final conclusion

Dataset: `C:\D\projects\Orca robot hand\orca sim\orca_sim\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`.

Corrected-Flex11 - Corrected-17 averaged across classifiers = +0.0196 accuracy.
OptimizedAction-Flex11 - OptimizedAction-17 averaged across classifiers = +0.0185 accuracy.
JointAngle-11 - Corrected-Flex11 averaged across classifiers = -0.0228 accuracy.
JointAngle-11 - OptimizedAction-Flex11 averaged across classifiers = -0.0141 accuracy.
Corrected-Flex11 - Corrected-PCA11 averaged across classifiers = +0.0087 accuracy.
OptimizedAction-Flex11 - OptimizedAction-PCA11 averaged across classifiers = +0.0120 accuracy.

Interpret these differences together with classifier-specific confidence intervals and paired Wilcoxon tests in `classifier_results_all.csv` and `paired_comparisons.csv`; no single classifier is treated as decisive.
