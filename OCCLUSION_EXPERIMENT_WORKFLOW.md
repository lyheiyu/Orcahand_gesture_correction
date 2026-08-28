# Synthetic Occlusion Experiment

This workflow creates landmark-level synthetic occlusion proxies from the clean six-class dataset.
It does not reproduce the exact response of MediaPipe to image-space occlusion because the source
videos are not available. The clean landmark coordinates remain the reference trajectory.

## Corruption model

For every sequence, the generator selects a contiguous temporal window and one or two fingers.
Only distal finger joints are changed so that the wrist and palm-scale anchors remain intact.

- `freeze`: hold the selected joints near their last visible coordinates;
- `drift`: hold the selected joints while adding a smooth coordinate drift;
- `collapse`: pull the selected joints toward the palm center;
- `mixed`: choose one of the three modes independently for every sequence.

Severity controls the window duration, number of fingers, and coordinate-noise scale:

| Severity | Window | Fingers | Noise scale |
|---|---:|---:|---:|
| Light | 15% | 1 | 0.02 |
| Medium | 30% | 1 | 0.05 |
| Heavy | 45% | 2 | 0.08 |

## Smoke test

Generate two sequences first:

```powershell
python .\generate_occlusion_dataset.py `
  --input .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --output .\diagnostics\occlusion_smoke_20260826\occluded_medium_base.csv `
  --manifest .\diagnostics\occlusion_smoke_20260826\occlusion_manifest.csv `
  --severity medium `
  --mode mixed `
  --seed 42 `
  --version v2 `
  --max-sequences 2
```

Regenerate the optimized representations from the corrupted observations:

```powershell
python .\augment_dataset_with_optimization.py `
  --input .\diagnostics\occlusion_smoke_20260826\occluded_medium_base.csv `
  --output .\diagnostics\occlusion_smoke_20260826\occluded_medium_optimized.csv `
  --version v2
```

## Full medium-severity dataset

Omit `--max-sequences` to process all 571 sequences:

```powershell
python .\generate_occlusion_dataset.py `
  --input .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --output .\diagnostics\occlusion_20260826\occluded_medium_base.csv `
  --manifest .\diagnostics\occlusion_20260826\occlusion_medium_manifest.csv `
  --severity medium `
  --mode mixed `
  --seed 42 `
  --version v2

python .\augment_dataset_with_optimization.py `
  --input .\diagnostics\occlusion_20260826\occluded_medium_base.csv `
  --output .\diagnostics\occlusion_20260826\occluded_medium_optimized.csv `
  --version v2
```

The optimization step is substantially slower than coordinate corruption because it runs the
MuJoCo fitting procedure for every frame.

## Evaluation rule

For the robustness experiment, fit the classifier and all preprocessing operations on clean
training sequences, then evaluate the same fitted model on the corresponding corrupted test
sequences. Do not fit the scaler, PCA, or classifier on corrupted test data. Use the same sequence
IDs and split manifest for every representation and severity.

Recommended outputs are classification degradation, actuator recovery error, landmark recovery
error, spike suppression, motion-amplitude retention, and temporal lag. Report clean, light,
medium, and heavy conditions separately.

Run the matched clean-to-occluded classification evaluation with:

```powershell
python .\evaluate_occlusion_robustness.py `
  --clean .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --occluded .\diagnostics\occlusion_smoke_20260826\occluded_medium_optimized.csv `
  --output-dir .\figures\occlusion_robustness_6class_20260826 `
  --shots 3 `
  --repeats 20 `
  --random-state 42
```

The evaluator fits each classifier, StandardScaler, and PCA transformation using clean training
sequences only. Every fitted model is then evaluated on both the clean test sequences and their
matched occluded versions.
