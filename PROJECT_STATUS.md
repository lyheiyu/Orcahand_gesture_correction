# Project Status

## Project Goal

This project explores whether MediaPipe hand landmarks can be improved by projecting them into an ORCA hand state space, and whether the corrected representation can improve gesture classification under low-data settings.

The long-term goal is to move from:

- raw MediaPipe landmarks

to:

- ORCA-aware corrected features
- MuJoCo-based optimized features
- temporally consistent sequence features


## What Has Been Built

### 1. MediaPipe Teleoperation Prototype

Implemented in:

- [mediapipe_teleop.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/mediapipe_teleop.py)

Features:

- webcam-based MediaPipe hand tracking
- ORCA right-hand control
- mirror handling
- right-hand selection
- base yaw / pitch / roll teleoperation
- neutral-pose calibration with `c`

This prototype is useful for understanding the mapping, but it is not the final research contribution.


### 2. Frame-Wise Feature Pipeline

Implemented in:

- [gesture_features.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/src/orca_sim/gesture_features.py)

Current feature groups:

- `raw`
  - normalized MediaPipe landmarks
- `geom`
  - hand-crafted geometric features
- `corrected`
  - ORCA-aware actuator-space projection with embodiment-aware bounds

Interpretation:

- `raw` is the visual baseline
- `geom` is the geometric baseline
- `corrected` is the structure-aware ORCA representation


### 3. Dataset Collection

Implemented in:

- [collect_gesture_dataset.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/collect_gesture_dataset.py)

Supports:

- single-frame collection
- sequence collection with `--sequence-mode`
- `sequence_id`
- `frame_id`
- `timestamp_sec`

This means:

- frame-wise experiments can be done with isolated samples
- sequence experiments can be done by grouping rows by `sequence_id`


### 4. Few-Shot Classification Baseline

Implemented in:

- [train_svm.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/train_svm.py)

Supports:

- `raw / geom / corrected / all`
- `shots-per-class`
- repeated evaluation with `mean +- std`
- sequence aggregation with `--sequence-mode`

Current conclusion from frame-wise few-shot experiments:

- `corrected` performed better than `raw`
- this suggests the ORCA-aware representation is useful in low-data classification


### 5. MuJoCo Optimization Prototype

Implemented in:

- [mujoco_optimizer.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/src/orca_sim/mujoco_optimizer.py)
- [fit_mediapipe_frame.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/fit_mediapipe_frame.py)

This is the first optimization-based prototype that:

- uses ORCA actuator states as optimization variables
- uses MuJoCo forward kinematics through `mj_forward`
- fits sparse hand keypoints
- includes palm-orientation alignment
- regularizes toward heuristic ORCA projection
- regularizes toward the ORCA default pose
- includes boundary penalties

Important:

- this is stronger than simple heuristic projection
- but it is still a prototype, not yet a complete physics-based reconstruction system


## Current Scientific Position

At this stage, the safest description is:

- ORCA-aware structural correction
- MuJoCo-based ORCA-constrained pose fitting prototype

Avoid overstating it as:

- fully physical reconstruction
- exact physics-based optimization

The current optimization uses MuJoCo as a forward model and embodiment-constrained state space, but it is not yet a full contact-aware or differentiable-physics method.


## Main Findings So Far

### Frame-Wise Few-Shot

In previous few-shot experiments on frame-wise data:

- `corrected` outperformed `raw`
- this is the first evidence that ORCA-aware correction can improve classification in low-data settings

### Sequence-Level Classification

Sequence collection and sequence-level training now work.

At the moment:

- sequence datasets are still small
- test sets are still tiny
- reported variance is still high

So the sequence results are promising but not yet stable enough for final claims.

### MuJoCo Optimization

The optimization prototype successfully runs and converges.

Observed behavior:

- it reduces the purely heuristic nature of the correction
- regularization now prevents extreme boundary collapse better than the first version
- some actuator dimensions, especially wrist, can still be pushed toward limits

This means the optimization is working, but the objective design still needs refinement.


## Current Limitations

1. Dataset size is still small for a research paper.
2. Sequence datasets need more clips per class.
3. Labels are not yet fully standardized across all datasets.
4. The optimization objective still uses sparse correspondences and simple priors.
5. `optimized` features are not yet integrated into the training pipeline.


## Immediate Next Steps

### Priority 1: Standardize the Dataset

- use one consistent label convention
- for example always use:
  - `6`
  - `7`
  - `8`
- avoid mixing `six` and `6`, `eight` and `8`

### Priority 2: Expand Sequence Collection

Target:

- at least 10 sequences per class
- preferably 15 to 20 sequences per class

Suggested classes for next round:

- `6`
- `7`
- `8`
- optionally add one or two visually similar classes later

### Priority 3: Integrate `optimized` Features

Next major code task:

- run the MuJoCo optimizer during feature export
- save a new feature group:
  - `optimized_*`
- compare:
  - `raw`
  - `geom`
  - `corrected`
  - `optimized`

### Priority 4: Multi-Model Comparison

For paper-quality experiments, extend beyond SVM:

- KNN
- SVM
- MLP
- RandomForest

### Priority 5: Better Evaluation Protocol

Add:

- `mean +- std`
- macro F1
- confusion matrix
- sequence-level evaluation
- later: cross-session split


## Recommended Weekend Workflow

1. Install dependencies

```powershell
conda activate orca
cd "C:\D\projects\Orca robot hand\orca sim\orca_sim"
python -m pip install -r .\requirements-dev.txt
python -m pip install -e .
```

2. Continue collecting sequences

```powershell
python .\collect_gesture_dataset.py --label 6 --output gesture_sequence_dataset.csv --hand-landmarker-model ".\hand_landmarker.task" --target-hand right --sequence-mode
python .\collect_gesture_dataset.py --label 7 --output gesture_sequence_dataset.csv --hand-landmarker-model ".\hand_landmarker.task" --target-hand right --sequence-mode
python .\collect_gesture_dataset.py --label 8 --output gesture_sequence_dataset.csv --hand-landmarker-model ".\hand_landmarker.task" --target-hand right --sequence-mode
```

3. Test the MuJoCo optimization prototype

```powershell
python .\fit_mediapipe_frame.py --dataset .\gesture_sequence_dataset.csv --label 6
```

4. After enough sequence data is collected, compare sequence-level features

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set geom --sequence-mode --shots-per-class 3 --repeats 20
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set corrected --sequence-mode --shots-per-class 3 --repeats 20
```


## Paper Direction

This project is now at the stage of a promising research prototype.

A realistic paper framing is:

- MediaPipe landmarks are noisy under low-data gesture recognition settings
- ORCA-aware corrected features improve few-shot classification over raw landmarks
- MuJoCo-based ORCA-constrained optimization provides a stronger correction mechanism
- sequence-aware correction is the next stage of the method

This is strong enough to continue toward a research paper, but it still needs:

- more data
- stronger sequence experiments
- optimized-feature integration
- multi-model comparison


## Key Files

- [gesture_features.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/src/orca_sim/gesture_features.py)
- [mujoco_optimizer.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/src/orca_sim/mujoco_optimizer.py)
- [collect_gesture_dataset.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/collect_gesture_dataset.py)
- [train_svm.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/train_svm.py)
- [fit_mediapipe_frame.py](c:/D/projects/Orca robot hand/orca sim/orca_sim/fit_mediapipe_frame.py)
- [MUJOCO_OPTIMIZATION_METHOD.md](c:/D/projects/Orca robot hand/orca sim/orca_sim/MUJOCO_OPTIMIZATION_METHOD.md)
- [SEQUENCE_RESEARCH_PLAN.md](c:/D/projects/Orca robot hand/orca sim/orca_sim/SEQUENCE_RESEARCH_PLAN.md)
- [requirements-dev.txt](c:/D/projects/Orca robot hand/orca sim/orca_sim/requirements-dev.txt)
