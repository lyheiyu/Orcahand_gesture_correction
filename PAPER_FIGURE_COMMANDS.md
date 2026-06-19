# Paper Figure Commands

This file collects the main commands for generating the tables and figures used in the paper.

All commands assume the current working directory is:

```powershell
C:\D\projects\Orca robot hand\orca sim\orca_sim
```

Recommended environment:

```powershell
conda activate orca
```

## 0. Demo / Teleop / Data Collection

This section collects the most commonly used runtime commands for live demos,
teleoperation, and dataset collection.

### 0.1 Main MediaPipe Teleop Demo

Current recommended command for the palm-facing right-hand demo:

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --camera-side palm --control-space display --hand-landmarker-model ".\hand_landmarker.task" --disable-auto-base --smoothing 0.10
```

Meaning of the most important options:

- `--target-hand right`
  - drive ORCA from the detected right hand
- `--camera-side palm`
  - use the palm-facing-camera convention
- `--control-space display`
  - control ORCA from the displayed mirrored image space
- `--disable-auto-base`
  - keep the base fixed during the hand-only demo
- `--smoothing 0.10`
  - apply light action smoothing for a more stable demo

### 0.2 Back-of-Hand Teleop Variant

If the camera mainly sees the back of the hand instead of the palm:

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --camera-side back --control-space display --hand-landmarker-model ".\hand_landmarker.task" --disable-auto-base --smoothing 0.10
```

### 0.3 Teleop with Base Motion Enabled

If you want the teleop scene to also use the base yaw / pitch / roll mapping:

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --camera-side palm --control-space display --hand-landmarker-model ".\hand_landmarker.task" --smoothing 0.10
```

Note:

- omit `--disable-auto-base`
- current base control is still heuristic and is better for demos than for quantitative evaluation

### 0.4 Collect a New Gesture Sequence

Example command for collecting one new optimized sequence and exporting the v2
feature groups:

```powershell
python .\collect_gesture_dataset.py --label 8 --output gesture_sequence_dataset_more.csv --hand-landmarker-model ".\hand_landmarker.task" --target-hand right --sequence-mode --export-optimized --version v2
```

Typical usage:

- change `--label` to the gesture class you are recording
- change `--output` if you want to store the new capture in a different CSV

### 0.5 Merge / Continue Using Datasets

Main current optimized dataset for experiments:

```powershell
gesture_sequence_dataset_optimized_v2.csv
```

Smoothing-augmented dataset:

```powershell
gesture_sequence_dataset_with_smoothing.csv
```

If you collect additional sequences first and want them reflected in later
classification or figure-generation runs, make sure the downstream commands point
to the updated dataset file you actually want to evaluate.

## 1. Generate Smoothing Baseline Dataset

This creates `gesture_sequence_dataset_with_smoothing.csv` from the raw landmark columns in `gesture_sequence_dataset_optimized_v2.csv`.

```powershell
python .\generate_smoothing_baselines.py --input .\gesture_sequence_dataset_optimized_v2.csv --output .\gesture_sequence_dataset_with_smoothing.csv
```

Generated feature groups:

- `moving_average_raw`
- `savgol_raw`
- `oneeuro_raw`
- `kalman_raw`

## 2. Main Structured-Representation Results Across Classifiers

This is the main four-classifier comparison for:

- `raw`
- `corrected`
- `optimized_action`
- `optimized_full`

The easiest way is to use the batch script:

```powershell
python .\generate_classifier_figures.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --output-dir .\figures\classifier_suite_v2 --results-csv classification_suite_v2.csv --sequence-mode
```

Main outputs:

- `figures\classifier_suite_v2\classification_suite_v2.csv`
- `figures\classifier_suite_v2\classification_svm.png`
- `figures\classifier_suite_v2\classification_knn.png`
- `figures\classifier_suite_v2\classification_rf.png`
- `figures\classifier_suite_v2\classification_mlp.png`
- `figures\classifier_suite_v2\cm_*.png`

## 3. PCA Sweep Results

This generates PCA baselines and comparison figures.

```powershell
python .\generate_pca_sweep.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --output-dir .\figures\pca_sweep_v2 --sequence-mode --shots-per-class 3 --repeats 20
```

Main outputs:

- `figures\pca_sweep_v2\pca_sweep_results.csv`
- `figures\pca_sweep_v2\pca_sweep_summary.csv`
- `figures\pca_sweep_v2\*.png`

## 4. Jitter / Stability Evaluation

This evaluates temporal stability for the current main feature groups.

```powershell
python .\evaluate_jitter.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --feature-sets raw corrected optimized_action optimized_full --results-csv .\figures\jitter_v2.csv --plot .\figures\jitter_v2.png
```

Main outputs:

- `figures\jitter_v2.csv`
- `figures\jitter_v2.png`

## 5. RandomForest Smoothing Comparison

These commands generate the main landmark-space smoothing comparison used in the paper.

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set moving_average_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set savgol_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set oneeuro_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set kalman_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf
```

Reference comparison rows:

- `raw` comes from the standard structured results
- `optimized_full` comes from the standard structured results

## 6. Build Full Smoothing Suite Across Four Classifiers

This section creates one combined CSV for:

- `raw`
- `moving_average_raw`
- `savgol_raw`
- `oneeuro_raw`
- `kalman_raw`
- `optimized_full`

across:

- `svm`
- `knn`
- `rf`
- `mlp`

Output file:

- `figures\smoothing_suite.csv`

### 6.1 SVM

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set moving_average_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set savgol_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set oneeuro_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set kalman_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set optimized_full --sequence-mode --shots-per-class 3 --repeats 20 --classifier svm --results-csv .\figures\smoothing_suite.csv
```

### 6.2 KNN

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set moving_average_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set savgol_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set oneeuro_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set kalman_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set optimized_full --sequence-mode --shots-per-class 3 --repeats 20 --classifier knn --results-csv .\figures\smoothing_suite.csv
```

### 6.3 RF

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set moving_average_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set savgol_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set oneeuro_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set kalman_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set optimized_full --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --results-csv .\figures\smoothing_suite.csv
```

### 6.4 MLP

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set moving_average_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set savgol_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set oneeuro_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set kalman_raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
python .\train_svm.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --feature-set optimized_full --sequence-mode --shots-per-class 3 --repeats 20 --classifier mlp --results-csv .\figures\smoothing_suite.csv
```

## 7. Best Confusion Matrices

Recommended main confusion matrices:

- `RF + optimized_action`
- `RF + raw_pca12`
- `RF + raw`

### 7.1 RF + Optimized Action

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --feature-set optimized_action --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --plot-confusion .\figures\cm_rf_optimized_action.png --confusion-title "RF - optimized_action"
```

### 7.2 RF + Best PCA

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --feature-set raw --pca-components 12 --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --plot-confusion .\figures\cm_rf_raw_pca12.png --confusion-title "RF - raw_pca12"
```

### 7.3 RF + Raw

```powershell
python .\train_svm.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --feature-set raw --sequence-mode --shots-per-class 3 --repeats 20 --classifier rf --plot-confusion .\figures\cm_rf_raw.png --confusion-title "RF - raw"
```

## 8. Generate Paper Summary Figures

This script collects the most important outputs into `figures\paper_summary`.

If you only want the main paper figures:

```powershell
python .\generate_paper_figures.py
```

If you also want the appendix-level smoothing-across-classifiers figure:

```powershell
python .\generate_paper_figures.py --smoothing-suite-csv .\figures\smoothing_suite.csv
```

Main outputs in `figures\paper_summary`:

- `smoothing_comparison.png`
- `representation_comparison.png`
- `classifier_accuracy_structured.png`
- `jitter_actuator_space.png`
- `jitter_landmark_space.png`
- `appendix_pca_across_classifiers.png`
- `appendix_smoothing_across_classifiers.png`
- `best_confusion_matrix_manifest.csv`

## 8.5 Automatic Experiment Suite

If you want one script that automatically:

- reads the dataset
- detects which feature sets actually exist
- runs multiple classifiers
- saves per-classifier summary plots
- saves confusion matrices

use:

```powershell
python .\generate_experiment_suite.py --dataset .\gesture_sequence_dataset_optimized_v2.csv --output-dir .\figures\auto_structured --sequence-mode --include-structured --include-pca17 --include-best-pca
```

For the smoothing dataset:

```powershell
python .\generate_experiment_suite.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --output-dir .\figures\auto_smoothing --sequence-mode --include-smoothing
```

For fully automatic detection of all feature groups present in the dataset:

```powershell
python .\generate_experiment_suite.py --dataset .\gesture_sequence_dataset_with_smoothing.csv --output-dir .\figures\auto_all --sequence-mode --include-all-detected
```

Outputs include:

- `experiment_results.csv`
- `classification_svm.png`
- `classification_knn.png`
- `classification_rf.png`
- `classification_mlp.png`
- `cms\cm_*.png`
- `run_manifest.csv`

## 9. Suggested Paper Usage

### Main paper

- `smoothing_comparison.png`
- `representation_comparison.png`
- `classifier_accuracy_structured.png`
- `jitter_actuator_space.png`
- `jitter_landmark_space.png`
- `best_cm_optimized_action.png`
- `best_cm_best_pca.png`

### Appendix / Supplementary

- `appendix_pca_across_classifiers.png`
- `appendix_smoothing_across_classifiers.png`
- `best_cm_raw.png`
- additional classifier-specific confusion matrices
