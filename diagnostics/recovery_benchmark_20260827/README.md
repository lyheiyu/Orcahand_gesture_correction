# Recovery Benchmark Workflow

This directory contains a controlled clean-reference trajectory recovery benchmark. The source MediaPipe trajectory is a reference observation, not physical ground truth.

## Design

- Source: `diagnostics/updated_6class_20260820/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`
- Statistical unit: one sequence/corruption instance
- Assignment: one balanced condition per sequence
- Seed pool: ten deterministic seeds beginning at 42
- Gaussian noise: standard deviations 0.01, 0.03, and 0.06
- Spike corruption: displacement magnitude 0.75 for 1, 2, or 3 frames
- Dropout: freeze distal thumb, index, or middle landmarks for 3 or 5 frames
- Methods: corrupted input, Kalman, One-Euro, Corrected, and unchanged Fixed OA

The local dropout does not corrupt MCP normalization anchors. Exact affected frames and landmarks are recorded in `corruption_manifest.csv`.

## Generate Corruptions

```powershell
python .\generate_recovery_corruptions.py `
  --input .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --output-dir .\diagnostics\recovery_benchmark_20260827 `
  --seed 42
```

## Run Benchmark

```powershell
python .\run_recovery_benchmark.py `
  --clean .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --corrupted .\diagnostics\recovery_benchmark_20260827\corrupted_landmarks.csv `
  --manifest .\diagnostics\recovery_benchmark_20260827\corruption_manifest.csv `
  --scenarios .\diagnostics\recovery_benchmark_20260827\scenario_manifest.csv `
  --output-dir .\diagnostics\recovery_benchmark_20260827\results `
  --version v2
```

## Main Outputs

- `results/RECOVERY_BENCHMARK_REPORT.md`: interpretation and Q1-Q6 decisions
- `results/per_sequence_landmark_metrics.csv`: primary paired landmark results
- `results/per_sequence_actuator_metrics.csv`: actuator-space corruption sensitivity
- `results/paired_wilcoxon_tests.csv`: landmark-space paired tests
- `results/actuator_paired_wilcoxon_tests.csv`: OA versus Corrected actuator tests
- `results/figures/`: fourteen recovery and motion-fidelity figures
- `corruption_config.json`: exact experiment configuration and Git commit

## Current Scope

The current result supports robust actuator-space refinement relative to Corrected. It does not support superior landmark recovery relative to Kalman or One-Euro filtering, and it must not be described as anatomical ground-truth recovery.
