# Frozen Confusion Matrices

These confusion matrices are regenerated from the frozen 456/115 development/final manifests with the same 3-shot, 20-repeat protocol used by the paper.

## Contents

- `cm_<classifier>_<representation>.png/pdf`: eight individual matrices.
- `paired_cm_<classifier>.png/pdf`: JointAngle-11 and Compact Refined-7 side by side for each classifier.
- `all_classifiers_confusion_overview.png/pdf`: all eight matrices in one supplementary figure.
- `aggregate_confusion_matrices.csv`: counts and row-normalized values.
- `frozen_final_predictions.csv`: all 18,400 repeat-level final predictions.
- `reproduction_validation.csv`: comparison against the historical frozen metric file.
- `confusion_matrix_metadata.json`: dataset and manifest hashes and protocol settings.
- `cm_stability_statistics.xlsx`: per-cell mean, sample std, 95% CI, minimum, and maximum.
- `stability_mean_std/`: eight CM figures annotated as mean $\pm$ sample std.
- `per_class_recall_stability.xlsx`: focused diagonal-recall mean/std and paired differences.
- `recall_stability/`: four classifier figures and one overview using per-class recall mean $\pm$ std.

Each cell displays row-normalized frequency and cumulative count in parentheses. Counts sum predictions across 20 repeats on the same 115 final sequences; they are descriptive repeated predictions, not 2,300 independent test samples.

The current environment reproduces 159 of 160 classifier-representation-repeat metric groups within `1e-12`. The only mismatch is RandomForest + Compact Refined-7 at repeat 16, where the current environment classifies one additional sequence correctly. This likely reflects a software/environment difference in RandomForest. SVM, KNN, MLP, and all JointAngle-11 groups reproduce exactly.

The CM standard deviation measures sensitivity to the repeated 3-shot training selection. It is not actuator trajectory stability and does not represent variation across independent test cohorts. Diagonal cells are per-class recall mean/std; off-diagonal cells describe the stability of specific error routes.

## Regenerate

```powershell
python .\generate_frozen_confusion_matrices.py
```

Generate the Excel workbook and mean/std CM figures with a Python environment containing `openpyxl`:

```powershell
C:\Users\31734\anaconda3\python.exe .\export_cm_stability_excel.py
```

Optional explicit command:

```powershell
python .\generate_frozen_confusion_matrices.py `
  --dataset .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --frozen-dir .\diagnostics\orca_compact_selection_20260827 `
  --output-dir .\paper_final_compact_orca_20260827\figures\supplementary_confusion_matrices `
  --main-figure .\paper_final_compact_orca_20260827\figures\figure_10_main_svm_confusion_comparison `
  --shot 3 `
  --repeats 20 `
  --seed 42
```
