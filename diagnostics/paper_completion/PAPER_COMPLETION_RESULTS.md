# Paper Completion Experiments

## Frozen protocol

- Dataset: `diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`
- 56 sequences, 7109 frames, 6 classes
- 3 training sequences per class
- 20 repeated sequence-level holdouts
- Identical sequence IDs for every representation and classifier
- PCA is fit only on training frames in each repeat

The CSV does not contain `subject_id` or `session_id`. These results are sequence-disjoint, not verified subject-independent results.

## PCA-17 across classifiers

| Representation | SVM | KNN | RF | MLP |
|---|---:|---:|---:|---:|
| Raw | 0.8250 | 0.7792 | 0.8708 | 0.8208 |
| PCA-17 | 0.8958 | 0.9125 | 0.9458 | 0.9167 |
| Corrected | 0.9458 | 0.9000 | 0.9083 | 0.8958 |
| Optimized Action | **0.9667** | **0.9250** | 0.9292 | 0.9125 |
| Optimized Full | 0.7917 | 0.7500 | 0.8500 | 0.7833 |

Interpretation: Optimized Action is strongest under SVM and KNN. PCA-17 is strongest under RandomForest and narrowly strongest under MLP. The method is a competitive, interpretable structured representation, not a universally superior dimensionality-reduction method.

## Loss ablation

| Variant | Velocity mean | Acceleration mean | SVM accuracy | RF accuracy |
|---|---:|---:|---:|---:|
| Corrected only | 0.5601 | 0.8687 | 0.9458 | 0.9083 |
| Full | 0.3015 | 0.3206 | 0.9667 | 0.9292 |
| Without palm normal | 0.2931 | 0.3167 | 0.9458 | 0.9125 |
| Without acceleration | 0.3254 | 0.4467 | 0.9708 | 0.9375 |
| Without temporal terms | 0.3900 | 0.6008 | 0.9667 | 0.9375 |
| L2 instead of Huber | 0.2765 | 0.3160 | 0.9458 | 0.9250 |

Confirmed interpretation:

- Temporal and acceleration terms substantially reduce temporal variation.
- Their contribution is stability, not a guaranteed independent accuracy gain.
- Palm-normal alignment improves SVM accuracy by 0.0208 relative to its ablation; the exploratory paired Wilcoxon value is 0.0384.
- L2 does not improve classification and increases mean solve time to about 72.9 ms/frame.
- Repeated holdout splits overlap, so all Wilcoxon values are exploratory.

## Runtime

The fixed-weight full optimizer was benchmarked over 300 consecutive frames after one warm-up solve.

- CPU: Intel Core i9-14900K
- Python 3.11.15
- MuJoCo 3.6.0
- SciPy 1.17.1
- NumPy 2.4.4
- Mean: 29.93 ms/frame
- Median: 29.21 ms/frame
- P95: 37.28 ms/frame
- Mean iterations: 5.87
- Success rate: 100%
- Finite-output rate: 100%

The implementation is causal, but the P95 latency does not support a strict 30-FPS real-time claim.

## Reproduction

```powershell
python .\run_paper_completion_experiments.py `
  --dataset .\diagnostics\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --split-manifest .\diagnostics\palm_fix_split_manifest_6class.csv `
  --workers 4 --runtime-max-frames 300
```

Use `--force-ablation` only when the optimizer or ablation definitions change. Figure-only and paper edits should reuse the cached frame-level ablation files.
