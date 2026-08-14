# GPT Handoff For Paper

## 1. Paper Identity

### Current title

MuJoCo-Constrained Temporal Refinement of Hand Landmark Sequences for Few-Shot Gesture Recognition

### Short positioning

This paper is not mainly about inventing a new temporal classifier.

It is about:

- refining noisy MediaPipe hand landmark sequences
- using ORCA actuator-space structure
- constraining the refinement with MuJoCo forward kinematics
- evaluating whether the refined representation improves downstream few-shot gesture recognition

### One-sentence contribution

We refine noisy MediaPipe hand landmarks in an ORCA actuator space using MuJoCo-constrained temporal optimization, and show that the resulting latent representation is more useful for few-shot gesture recognition than raw landmarks and heuristic actuator projections.

## 2. Core Motivation

MediaPipe hand landmarks are convenient, but under self-occlusion, oblique viewpoints, and rapid finger motion, the landmark sequence can contain:

- jitter
- drift
- transient outliers
- anatomically implausible variations

Many existing pipelines directly feed these landmarks into temporal classifiers. This paper instead asks:

Can we improve the landmark sequence itself before downstream recognition?

The main motivation is:

- coordinate-space smoothing may reduce jitter
- but smoothing alone may not preserve the most discriminative hand articulation structure
- a structured actuator-space representation may be more robust and interpretable

## 3. Method Overview

### Input

- MediaPipe 21x3 hand landmarks per frame
- vectorized as 63-dimensional raw feature

### Representation pipeline

1. Raw landmarks
2. Heuristic actuator-space projection
3. MuJoCo-constrained temporal refinement in ORCA actuator space
4. Optional re-projection back to landmark space
5. Sequence-level statistical aggregation for few-shot classification

### Representations used in experiments

- Raw
  - 63D MediaPipe landmarks
- Raw PCA
  - PCA-compressed raw landmarks
- Corrected
  - 17D heuristic actuator-space projection
- Optimized Action
  - 17D refined actuator-space latent representation
- Optimized Full
  - 63D reprojected landmarks from the optimized actuator state

## 4. Mathematical Formulation

Let:

- y_t in R^(21x3): MediaPipe landmarks at frame t
- q_t in R^17: ORCA actuator-space latent state
- h(q_t): MuJoCo forward kinematic mapping from actuator state to landmarks

The optimization target is:

q_t^* = arg min over q_t in Q of

- landmark fitting term
- palm-normal alignment term
- prior term from heuristic actuator projection
- first-order temporal term
- second-order acceleration term
- default-pose regularization
- actuator-boundary regularization

The most important temporal regularizer is:

L_acceleration = || q_t - 2 q_(t-1) + q_(t-2) ||_2^2

The landmark fitting term uses a Huber loss to reduce sensitivity to transient outliers.

## 5. Why This Is Not Just Smoothing

This is an important point for writing and discussion.

The method is not just applying a filter to landmark coordinates.

Differences from ordinary smoothing:

- smoothing operates directly in landmark space
- this method operates in ORCA actuator space
- candidate states are constrained by MuJoCo forward kinematics
- the latent representation is structurally meaningful and lower-dimensional
- the goal is not only lower jitter, but a better downstream representation

Important interpretation:

- the smoothest landmark sequence is not necessarily the best representation for classification
- coordinate smoothness and discriminative quality are related but different

## 6. Dataset Used In Current Main Paper

### Main dataset

gesture_sequence_dataset_chinese_dance_6class.csv

### Gesture classes

- orchid_palm
- orchid_finger
- flower_pinch
- prayer_beads
- three_finger_bent
- deer_horn

### Current scale

- 56 sequences
- 7109 frames

### Per-class counts

- orchid_palm: 7 sequences, 1064 frames
- orchid_finger: 11 sequences, 1323 frames
- flower_pinch: 12 sequences, 1770 frames
- prayer_beads: 12 sequences, 1314 frames
- three_finger_bent: 7 sequences, 805 frames
- deer_horn: 7 sequences, 833 frames

### Current evaluation setting

- few-shot sequence classification
- 3 training sequences per class
- 20 repeated splits

## 7. Sequence-Level Evaluation Protocol

This paper does not yet use GRU/LSTM as the main evaluation.

Instead, each sequence is converted into a fixed-length descriptor using summary statistics:

- mean
- standard deviation
- max
- delta = last frame minus first frame

If each frame feature is x_t in R^d, then the final sequence descriptor is:

z^(s) = [mean, std, max, delta] in R^(4d)

Reason for using this:

- keep downstream classifier lightweight
- make representation quality the main variable
- avoid a stronger temporal classifier dominating the story
- more stable for small-data few-shot experiments

## 8. Current Main Experimental Results

### 8.1 Main classifier comparison

Current six-class results:

#### SVM

- Raw: accuracy 0.8250 +- 0.0909, macro-F1 0.7985 +- 0.1187, kappa 0.7841 +- 0.1115
- Corrected: accuracy 0.9458 +- 0.0477, macro-F1 0.9262 +- 0.0759, kappa 0.9330 +- 0.0586
- Optimized Action: accuracy 0.9542 +- 0.0415, macro-F1 0.9378 +- 0.0713, kappa 0.9429 +- 0.0516
- Optimized Full: accuracy 0.7958 +- 0.1163, macro-F1 0.7334 +- 0.1534, kappa 0.7468 +- 0.1437

#### KNN

- Raw: accuracy 0.7792 +- 0.1325, macro-F1 0.7394 +- 0.1476, kappa 0.7319 +- 0.1577
- Corrected: accuracy 0.9000 +- 0.0677, macro-F1 0.9010 +- 0.0747, kappa 0.8781 +- 0.0819
- Optimized Action: accuracy 0.9083 +- 0.0786, macro-F1 0.9048 +- 0.0897, kappa 0.8881 +- 0.0955
- Optimized Full: accuracy 0.7542 +- 0.1330, macro-F1 0.7028 +- 0.1480, kappa 0.6998 +- 0.1601

#### RandomForest

- Raw: accuracy 0.8708 +- 0.0617, macro-F1 0.8588 +- 0.0826, kappa 0.8413 +- 0.0750
- Corrected: accuracy 0.9083 +- 0.0741, macro-F1 0.8850 +- 0.0982, kappa 0.8874 +- 0.0907
- Optimized Action: accuracy 0.9292 +- 0.0660, macro-F1 0.9168 +- 0.0866, kappa 0.9130 +- 0.0810
- Optimized Full: accuracy 0.8750 +- 0.0968, macro-F1 0.8676 +- 0.1218, kappa 0.8457 +- 0.1188

#### MLP

- Raw: accuracy 0.8208 +- 0.1030, macro-F1 0.7923 +- 0.1273, kappa 0.7820 +- 0.1220
- Corrected: accuracy 0.8958 +- 0.0447, macro-F1 0.8781 +- 0.0755, kappa 0.8720 +- 0.0549
- Optimized Action: accuracy 0.9167 +- 0.0646, macro-F1 0.9062 +- 0.0851, kappa 0.8977 +- 0.0791
- Optimized Full: accuracy 0.7708 +- 0.1288, macro-F1 0.7141 +- 0.1555, kappa 0.7162 +- 0.1585

### Main conclusion from classifier comparison

- Optimized Action is the strongest structured representation across all four classifiers
- Best overall structured result: SVM + Optimized Action

### 8.2 Smoothing baseline results

RandomForest smoothing comparison:

- Raw: accuracy 0.8708 +- 0.0617, macro-F1 0.8588 +- 0.0826, kappa 0.8413 +- 0.0750
- Moving Average: accuracy 0.8667 +- 0.0890, macro-F1 0.8488 +- 0.1139, kappa 0.8360 +- 0.1089
- Savitzky-Golay: accuracy 0.8667 +- 0.0808, macro-F1 0.8494 +- 0.1089, kappa 0.8362 +- 0.0984
- One-Euro: accuracy 0.8792 +- 0.0853, macro-F1 0.8646 +- 0.1104, kappa 0.8517 +- 0.1038
- Kalman: accuracy 0.9125 +- 0.0720, macro-F1 0.9064 +- 0.0854, kappa 0.8919 +- 0.0885
- Corrected: accuracy 0.9083 +- 0.0741, macro-F1 0.8850 +- 0.0982, kappa 0.8874 +- 0.0907
- Optimized Action: accuracy 0.9292 +- 0.0660, macro-F1 0.9168 +- 0.0866, kappa 0.9130 +- 0.0810
- Optimized Full: accuracy 0.8750 +- 0.0968, macro-F1 0.8676 +- 0.1218, kappa 0.8457 +- 0.1188

### Main conclusion from smoothing comparison

- Kalman is the strongest conventional smoothing baseline
- Kalman reduces landmark-space jitter more aggressively
- But Optimized Action gives the best RandomForest classification result
- Therefore the proposed benefit cannot be explained as simple coordinate smoothing

### 8.3 Jitter results

#### Actuator space

- Corrected: velocity mean 0.5601, velocity RMS 0.7903, acceleration mean 0.8687, acceleration RMS 1.1740
- Optimized Action: velocity mean 0.2964, velocity RMS 0.3849, acceleration mean 0.3198, acceleration RMS 0.4223

Conclusion:

- Optimized Action is clearly smoother than Corrected in actuator space

#### Landmark space

- Raw: velocity mean 0.5476, velocity RMS 0.7685, acceleration mean 0.7744, acceleration RMS 1.0372
- Moving Average: velocity mean 0.3299, velocity RMS 0.3908, acceleration mean 0.1648, acceleration RMS 0.2105
- Savitzky-Golay: velocity mean 0.3948, velocity RMS 0.4737, acceleration mean 0.2464, acceleration RMS 0.3006
- One-Euro: velocity mean 0.3054, velocity RMS 0.3828, acceleration mean 0.1892, acceleration RMS 0.2897
- Kalman: velocity mean 0.1769, velocity RMS 0.2112, acceleration mean 0.0742, acceleration RMS 0.1161
- Optimized Full: velocity mean 0.3052, velocity RMS 0.4170, acceleration mean 0.3120, acceleration RMS 0.4221

Conclusion:

- Ordinary smoothing reduces landmark-space jitter more than Optimized Full
- This is acceptable, because the paper’s main best representation is Optimized Action, not Optimized Full

### 8.4 PCA baseline

Focused PCA comparison:

- RandomForest + Raw: accuracy 0.8708, macro-F1 0.8588, kappa 0.8413
- RandomForest + Raw PCA-17: accuracy 0.9458, macro-F1 0.9528, kappa 0.9336
- RandomForest + Corrected: accuracy 0.9083, macro-F1 0.8850, kappa 0.8874
- RandomForest + Optimized Action: accuracy 0.9292, macro-F1 0.9168, kappa 0.9130
- SVM + Optimized Action: accuracy 0.9542, macro-F1 0.9378, kappa 0.9429

Main conclusion:

- PCA is a very strong low-dimensional baseline
- Optimized Action is still the best structured representation
- Best overall structured result is still SVM + Optimized Action
- The paper should not claim universal dominance over all PCA-based baselines in all classifier settings

## 9. Main Scientific Claims That Are Safe

These are the safest claims:

1. The proposed actuator-space refinement improves over raw landmarks and heuristic actuator projection across multiple lightweight classifiers.
2. The best structured representation is the optimized actuator latent state rather than the reconstructed full landmark sequence.
3. Conventional coordinate-space smoothing, especially Kalman filtering, can reduce landmark-space jitter more aggressively, but the best downstream recognition performance still comes from the refined actuator-space representation.
4. Strong PCA baselines show that generic low-dimensional geometric compression remains highly competitive, so the contribution should be framed as a structured and interpretable refinement approach rather than universal performance dominance.

## 10. Main Limitations

These limitations should be stated honestly:

- dataset is still relatively small
- dataset is moderately imbalanced
- likely limited subject diversity and session diversity
- sequence encoding is simple statistical aggregation, not a strong temporal model
- no ground-truth landmark correction benchmark
- current strongest PCA baseline is competitive

## 11. What Still Needs To Be Improved

### Highest priority

1. More data
   - more sequences per class
   - better class balance
   - more sessions
   - ideally more subjects

2. Clean final PDF
   - all citations resolved
   - table and figure ordering natural
   - no broken references

3. Final figure refresh
   - consistent export quality
   - final captions
   - one clean paper figure folder

### Nice to have

4. Statistical significance testing
   - especially comparing Optimized Action vs Kalman
   - and Optimized Action vs Raw PCA-17

5. Stronger temporal baselines
   - GRU
   - LSTM

6. Subject/session description
   - who recorded data
   - whether splits are subject-independent
   - how recording sessions differ

## 12. Figures Needed In The Paper

Current main figures:

1. Dataset distribution
2. Actuator-space jitter comparison
3. Landmark-space jitter comparison
4. Classifier comparison across SVM / KNN / RF / MLP
5. Smoothing baseline comparison
6. Representative confusion matrices
7. PCA comparison

## 13. Artifacts In The Project

Important files in the repo:

- main.tex
- references.bib
- figures/paper_rewrite_main/

Useful supporting CSV files:

- figures/paper_rewrite_main/main_results_6class.csv
- figures/paper_rewrite_main/smoothing_baseline_results_6class.csv
- figures/paper_rewrite_main/pca_baseline_results_6class.csv
- figures/paper_rewrite_main/dataset_summary_6class.csv
- figures/paper_rewrite_main/actuator_definition_table.csv
- figures/paper_rewrite_main/optimization_hyperparameters.csv

## 14. Prompt To Give Another GPT For Word Drafting

Use the following prompt:

---

Please help me write a polished academic paper draft in formal English, suitable for journal submission, based on the following study.

Paper title:
MuJoCo-Constrained Temporal Refinement of Hand Landmark Sequences for Few-Shot Gesture Recognition

Research goal:
This paper refines noisy MediaPipe hand landmark sequences in the actuator space of an ORCA robotic hand model using MuJoCo-constrained temporal optimization. The goal is not to propose a new temporal classifier, but to improve the hand representation itself before downstream few-shot gesture recognition.

Core ideas:
- MediaPipe landmarks may contain jitter, drift, and transient outliers
- A heuristic actuator-space projection gives a 17D structured representation
- A MuJoCo-constrained optimization refines this actuator representation with robust fitting, palm-normal alignment, prior regularization, temporal smoothness, acceleration regularization, default-pose regularization, and actuator-boundary penalties
- The refined latent actuator representation is called Optimized Action
- The reprojected landmark representation is called Optimized Full

Important interpretation:
- The method is not ordinary coordinate-space smoothing
- Conventional smoothing can reduce landmark-space jitter more aggressively
- But the refined actuator-space representation gives the strongest downstream recognition result among structured and smoothing baselines
- Strong PCA baselines remain highly competitive

Dataset:
- 6 Chinese dance gesture classes
- 56 sequences
- 7109 frames
- Classes: orchid_palm, orchid_finger, flower_pinch, prayer_beads, three_finger_bent, deer_horn

Main results:
- Best structured result: SVM + Optimized Action
- Accuracy 0.9542 +- 0.0415
- Macro-F1 0.9378 +- 0.0713
- Kappa 0.9429 +- 0.0516

Smoothing comparison under RandomForest:
- Kalman is the strongest smoothing baseline
- But Optimized Action still has the best RF classification result

PCA comparison:
- RandomForest + Raw PCA-17 is highly competitive
- Therefore the paper should not claim universal dominance over PCA

Safe conclusions:
- Optimized Action is the strongest structured representation
- The main value is structured, interpretable actuator-space refinement
- The smoothest coordinate sequence is not necessarily the best for recognition

Please write:
1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
6. Results
7. Discussion
8. Conclusion

Please keep the tone academically cautious, avoid overclaiming, and present the contribution as strong but nuanced.

---

## 15. What I Need From The Word Version

If producing a Word draft, it should contain:

- clean section structure
- polished academic English
- cautious claims
- figure placeholders with captions
- table placeholders with titles
- no exaggerated novelty claims
- explicit discussion of limitations

