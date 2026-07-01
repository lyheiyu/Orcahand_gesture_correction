# Paper Draft Readable Version

This file is a readable Markdown snapshot of the current `paper_draft.tex` content, organized for quick review.

## Title

**Robust Hand Landmark Correction via MuJoCo-Constrained Temporal Optimization**

## Abstract

MediaPipe hand landmarks are convenient to use, but in practice they can become unstable under self-occlusion, oblique viewpoints, and rapid finger motion. We study whether these noisy observations can be improved before downstream gesture recognition by refining them in the actuator space of an articulated ORCA hand model. The proposed method combines a rule-based actuator initialization with MuJoCo-constrained optimization, robust landmark fitting, palm-normal alignment, prior regularization, and first- and second-order temporal consistency terms. Unlike ordinary coordinate-space smoothing, the refinement is carried out in a structured actuator space and checked through MuJoCo forward kinematics. On the current six-class Chinese dance gesture subset, the refined actuator representation gives strong few-shot sequence classification results across several lightweight classifiers, reaching `0.9542` accuracy with SVM and `0.9167` with MLP. At the same time, a RandomForest comparison shows that PCA-reduced raw landmarks remain a strong baseline, reaching `0.9458` accuracy. Overall, the results suggest that actuator-space refinement provides a stable and interpretable representation, while low-dimensional compression of raw landmarks can retain complementary discriminative information.

## Introduction

MediaPipe provides a practical real-time pipeline for hand landmark detection and lightweight tracking. It is easy to deploy and works well enough for many interactive tasks, which is why it is often used as the front end in gesture-recognition systems. In our own experiments, however, the landmark sequence is not always stable. Under self-occlusion, oblique viewpoints, or fast finger motion, the predicted keypoints can jitter from frame to frame, drift away from the underlying pose, or briefly move into configurations that do not look anatomically reasonable.

This becomes a problem when the landmarks are used directly for dynamic gesture classification. The downstream model does not see the true hand state; it only sees a sequence of estimated coordinates. If those coordinates are noisy, the sequence representation can mix actual gesture information with artifacts from the visual front end.

Many MediaPipe-based pipelines concentrate on what happens after landmark extraction, for example by adding a temporal encoder or a sequence classifier. In this paper we look one step earlier. Our interest is not mainly in designing a new temporal classifier, but in asking whether the landmark sequence itself can be made more reliable before recognition.

More specifically, we ask whether the structural prior of the ORCA hand, together with the MuJoCo forward model, can be used to convert noisy landmark observations into a cleaner latent representation that is more useful for few-shot gesture recognition.

### Core Questions

- Are raw MediaPipe landmarks stable enough for sequence recognition?
- Does the ORCA actuator space provide a better low-dimensional representation?
- Can MuJoCo-constrained optimization reduce transient landmark drift?
- Does a smoother latent actuator trajectory improve downstream few-shot classification?

### Positioning

This paper should be read as a MuJoCo-constrained refinement framework for improving the robustness of MediaPipe landmark sequences, rather than as a new temporal gesture classifier.

### Contributions

- A structured landmark-refinement pipeline that maps noisy MediaPipe observations into the actuator space of an articulated ORCA hand model.
- A MuJoCo-constrained temporal optimization objective with robust landmark fitting, prior regularization, and acceleration-aware temporal consistency.
- Experimental evidence that the refined latent representation improves few-shot dynamic gesture classification relative to raw landmarks, heuristic actuator-space projection, and PCA-reduced baselines.

## Related Work

### Vision-based Hand Keypoint Estimation

Modern vision-based hand trackers provide convenient 2D/3D keypoints together with lightweight frame-to-frame tracking, but they still struggle in the cases that matter most for articulated hands: self-occlusion, severe perspective change, and fast local motion. In such cases, the predicted landmarks may look acceptable in a single frame while remaining unstable over time.

### Physical and Kinematic Constraints for Pose Correction

A common way to improve noisy pose observations is to introduce priors from kinematic structure, embodiment, inverse kinematics, or temporal consistency. Instead of smoothing coordinates directly, these approaches try to regularize the solution in a lower-dimensional configuration space where implausible states are easier to reject. Our method follows this idea in the actuator space of an embodied ORCA hand and checks candidate configurations through MuJoCo forward kinematics.

### Few-shot Gesture Recognition

Few-shot gesture recognition is especially sensitive to representation quality, because there is limited data to average out nuisance variation. Many landmark-based pipelines treat extracted keypoints as ready-to-use temporal features and concentrate on recurrent or transformer-style sequence models. Here the emphasis is earlier in the pipeline: improving the frame-level representation before temporal modeling.

## Method

### 1. Raw Hand Landmark Extraction

Each frame starts from normalized MediaPipe hand landmarks:

- `21 x 3` landmarks
- vectorized as `63D`

This is the `raw` representation.

### 2. Embodiment-aware Actuator-space Projection

The first structured representation is:

- `corrected`
- `17D`

It is obtained by a rule-based projection from landmarks to the ORCA actuator space. It is not learned and not optimized. It is an embodiment-constrained reparameterization.

### 3. MuJoCo-constrained Causal Temporal Refinement

The optimized latent actuator state is:

- `optimized_action`
- `17D`

It is estimated with:

- robust landmark fitting
- palm-normal alignment
- prior regularization
- temporal smoothness
- acceleration regularization
- default-pose / boundary terms

This is the main representation of the paper.

### 4. Reconstructed Landmark Output

The optimized actuator state can be projected back through MuJoCo:

- `optimized_full`
- `63D`

This is useful for visualization and geometric evaluation, but not always the best feature for classification.

### 5. Sequence-level Statistical Aggregation

For downstream evaluation, each sequence is summarized with:

- mean
- standard deviation
- max
- delta (`last - first`)

This is only the evaluation protocol, not the correction method itself.

## Experiments

### Main Dataset

Main task: six-class Chinese dance gesture subset.

Labels:

- `orchid_palm`
- `orchid_finger`
- `flower_pinch`
- `prayer_beads`
- `three_finger_bent`
- `deer_horn`

Few-shot setting:

- `3` training sequences per class
- `20` repeated runs

### Stability Evaluation

Temporal smoothness is evaluated separately in:

- actuator space
- landmark space

This separation is important because these spaces have different dimensions and scales.

### Actuator-space Stability

Lower is better.

| Representation | Velocity Mean | Velocity RMS | Acceleration Mean | Acceleration RMS |
|---|---:|---:|---:|---:|
| Corrected | 0.6215 | 0.8251 | 0.9509 | 1.2118 |
| Optimized Action | **0.3343** | **0.4190** | **0.3580** | **0.4517** |

Interpretation:

- `optimized_action` is clearly smoother than `corrected` in actuator space.
- The temporal regularization is doing real work here.

### Landmark-space Stability

Lower is better.

| Representation | Velocity Mean | Velocity RMS | Acceleration Mean | Acceleration RMS |
|---|---:|---:|---:|---:|
| Raw | **0.4439** | **0.5566** | 0.5952 | **0.7256** |
| Optimized Full | 0.4517 | 0.7190 | **0.5194** | 0.8169 |

Interpretation:

- Do not compare these numbers directly to actuator-space values.
- They live in different spaces.

## Main Classification Results

### Six-class Chinese Dance Subset

| Classifier | Representation | Accuracy | Macro-F1 | Kappa |
|---|---|---:|---:|---:|
| SVM | Raw | 0.8250 | 0.7985 | 0.7841 |
| SVM | Corrected | 0.9458 | 0.9262 | 0.9330 |
| SVM | Optimized Action | **0.9542** | **0.9378** | **0.9429** |
| SVM | Optimized Full | 0.7958 | 0.7334 | 0.7468 |
| RF | Raw | 0.8708 | 0.8588 | 0.8413 |
| RF | Corrected | 0.9083 | 0.8850 | 0.8874 |
| RF | Optimized Action | 0.9292 | 0.9168 | 0.9130 |
| RF | Optimized Full | 0.8750 | 0.8676 | 0.8457 |
| MLP | Raw | 0.8208 | 0.7923 | 0.7820 |
| MLP | Corrected | 0.8958 | 0.8781 | 0.8720 |
| MLP | Optimized Action | 0.9167 | 0.9062 | 0.8977 |
| MLP | Optimized Full | 0.7708 | 0.7141 | 0.7162 |

### Main Takeaway

- `optimized_action` is the strongest structured representation across the reported classifier comparisons.
- Best structured result: `SVM + optimized_action`
- `optimized_full` is not the best classification feature.

## PCA Baseline

One key question is whether the gain comes only from dimensionality reduction.

Strong PCA result:

- `raw_pca17` under RandomForest
- Accuracy: `0.9458`
- Macro-F1: `0.9528`
- Kappa: `0.9336`

### Interpretation

This is an important result:

- PCA is not a weak baseline.
- Low-dimensional compression alone already helps a lot.
- The final story is not “our method beats PCA everywhere.”
- A more honest conclusion is that `optimized_action` and `raw_pca17` preserve different useful cues.

## Smoothing Baselines

The draft compares against:

- moving average
- Savitzky-Golay
- One-Euro
- Kalman

Interpretation:

- Ordinary coordinate-space smoothing is genuinely strong, especially Kalman filtering.
- In landmark-space jitter evaluation, conventional smoothing can reduce velocity and acceleration more aggressively than `optimized_full`.
- However, the best downstream classification result is still obtained by `optimized_action`, not by the smoothest landmark-space baseline.
- The most useful conclusion is therefore not “our method is the smoothest filter,” but “a structure-constrained actuator representation is more useful for recognition than coordinate smoothness alone.”

## Discussion

### Why Frame Correction Helps

If frame-level landmarks are unstable, the sequence descriptor is contaminated before classification even begins. Refining the landmarks in actuator space reduces nuisance variation before the downstream classifier sees the data.

### Why `optimized_action` Often Beats `optimized_full`

`optimized_action` is:

- lower-dimensional
- structured
- semantically aligned with articulation

By contrast, `optimized_full` returns to a higher-dimensional coordinate space.

### Why PCA Is Still Strong

PCA removes redundancy and some noise while preserving major geometric variation. For fine-grained gestures, this can remain very effective. This does not weaken the paper. Instead, it makes the conclusion more careful and more believable.

### Limitations

- The current dataset is still moderately imbalanced.
- The current evaluation uses lightweight statistical sequence aggregation.
- Stronger temporal baselines such as GRU/LSTM are still future work.
- Some useful geometric information in raw landmarks may not yet be fully captured by the actuator mapping.

## Conclusion

The current draft supports the following conclusion:

The proposed MuJoCo-constrained refinement improves MediaPipe landmark sequences by operating in the ORCA actuator space rather than by smoothing coordinates directly. On the six-class Chinese dance subset, the refined actuator representation consistently improves over raw landmarks and heuristic actuator projection. At the same time, strong PCA baselines show that low-dimensional compression of raw landmarks remains an important comparison point. Taken together, the results support the value of improving landmark-sequence quality before downstream gesture recognition, while also suggesting that structured actuator refinement and generic geometric compression preserve complementary information.

## Figures Currently Referenced in `paper_draft.tex`

These are the figures currently used in the draft:

- `figures/paper_rewrite_main/jitter_actuator_space.png`
- `figures/paper_rewrite_main/jitter_landmark_space.png`
- `figures/paper_rewrite_main/representation_comparison.png`
- `figures/paper_rewrite_main/smoothing_comparison.png`
- `figures/paper_rewrite_main/best_cm_optimized_action.png`
- `figures/paper_rewrite_main/best_cm_best_pca.png`

## Notes

- This Markdown file is for readability only.
- The editable main paper file is still `paper_draft.tex`.
