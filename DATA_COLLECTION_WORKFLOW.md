# Data Collection Workflow

## Goal

This file summarizes the commands and steps for running the gesture data collection pipeline in this project.

Main entry script:

- `collect_gesture_dataset.py`


## 1. Enter The Project

```powershell
cd "C:\D\projects\Orca robot hand\orca sim\orca_sim"
```


## 2. Install Dependencies

Recommended first-time setup:

```powershell
python -m pip install -r .\requirements-dev.txt
python -m pip install -e .
```

If you want the camera + MediaPipe tools:

```powershell
python -m pip install -e ".[teleop]"
```


## 3. Optional Smoke Test

Check that the simulator can run before opening the camera workflow:

```powershell
python .\random_policy.py --env right --render-mode rgb_array --steps 5
```


## 4. Prepare The MediaPipe Model

If your MediaPipe installation uses the Tasks backend, keep `hand_landmarker.task` in the project root:

- `.\hand_landmarker.task`

Or pass it explicitly with:

- `--hand-landmarker-model ".\hand_landmarker.task"`


## 5. Single-Frame Collection

Use this mode if you want isolated labeled samples.

```powershell
python .\collect_gesture_dataset.py --label 6 --output gesture_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task"
```

Controls:

- `space`: save the current frame
- `q` or `Esc`: quit

What gets written:

- `label`
- `sequence_id`
- `frame_id`
- `timestamp_sec`
- `raw_*`
- `geom_*`
- `corrected_*`


## 6. Sequence Collection

Use this mode if you want continuous clips for temporal modeling.

```powershell
python .\collect_gesture_dataset.py --label 6 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
```

Controls:

- first `space`: start recording one sequence
- second `space`: stop recording that sequence
- `q` or `Esc`: quit

Useful notes:

- each recorded clip gets a `sequence_id`
- frames in the clip get increasing `frame_id`
- `timestamp_sec` is measured from the start of that sequence


## 7. Collect Multiple Gesture Labels

Example for labels `6`, `7`, and `8`:

```powershell
python .\collect_gesture_dataset.py --label 6 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
python .\collect_gesture_dataset.py --label 7 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
python .\collect_gesture_dataset.py --label 8 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
```

Recommendation:

- keep labels consistent
- do not mix `6` and `six`
- do not mix `8` and `eight`


## 8. Useful Flags

Common options for `collect_gesture_dataset.py`:

- `--label 6`
  assign the gesture label

- `--output gesture_sequence_dataset.csv`
  choose the CSV file

- `--target-hand right`
  only save the right hand

- `--target-hand left`
  only save the left hand

- `--target-hand either`
  save whichever hand is detected

- `--sequence-mode`
  record continuous sequences instead of isolated frames

- `--sequence-id my_seq_01`
  force a fixed sequence id

- `--save-every-n-frames 2`
  save one sample every 2 tracked frames

- `--camera-id 1`
  choose another camera device

- `--no-mirror`
  disable selfie-style mirroring

- `--hand-landmarker-model ".\hand_landmarker.task"`
  explicitly set the local MediaPipe model file


## 9. Recommended Collection Protocol

For your current MuJoCo + sequence research direction:

- use `--sequence-mode`
- keep one label format only, such as `6`, `7`, `8`
- collect at least 10 sequences per class
- prefer 15 to 20 sequences per class if possible
- include mild viewpoint and wrist-rotation variation
- keep train/test splits grouped by `sequence_id`


## 10. After Collection

You can run the current frame-wise MuJoCo fitting prototype on one class sample:

```powershell
python .\fit_mediapipe_frame.py --dataset .\gesture_sequence_dataset.csv --label 6
```

You can also run the current SVM baseline:

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set geom --sequence-mode --shots-per-class 3 --repeats 20
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set corrected --sequence-mode --shots-per-class 3 --repeats 20
```


## 11. Minimal End-To-End Workflow

```powershell
cd "C:\D\projects\Orca robot hand\orca sim\orca_sim"
python -m pip install -r .\requirements-dev.txt
python -m pip install -e .
python .\collect_gesture_dataset.py --label 6 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
python .\collect_gesture_dataset.py --label 7 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
python .\collect_gesture_dataset.py --label 8 --output gesture_sequence_dataset.csv --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --sequence-mode
python .\fit_mediapipe_frame.py --dataset .\gesture_sequence_dataset.csv --label 6
python .\train_svm.py --dataset .\gesture_sequence_dataset.csv --feature-set corrected --sequence-mode --shots-per-class 3 --repeats 20
```


## 12. Common Problems

If the camera does not open:

- try `--camera-id 1`
- close other apps using the webcam

If MediaPipe complains about a missing model:

- place `hand_landmarker.task` in the project root
- or pass `--hand-landmarker-model ".\hand_landmarker.task"`

If no samples are saved:

- make sure a hand is detected in the preview
- in sequence mode, confirm recording was started with `space`
- check that the selected hand matches `--target-hand`
