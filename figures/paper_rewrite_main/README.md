# Paper Rewrite Main Artifacts

This folder is the self-contained figure and result-artifact folder for `main.tex`.

## Figures Used In The Main Draft

- `dataset_distribution_6class.png`
- `jitter_actuator_6class.png`
- `jitter_landmark_6class.png`
- `classification_svm.png`
- `classification_rf.png`
- `classification_knn.png`
- `classification_mlp.png`
- `smoothing_baselines_6class.png`
- `cm_svm_raw.png`
- `cm_svm_optimized_action.png`
- `cm_mlp_raw.png`
- `cm_mlp_optimized_action.png`
- `pca_comparison_6class.png`

## Main Result CSVs

- `dataset_summary_6class.csv`
- `main_results_6class.csv`
- `smoothing_baseline_results_6class.csv`
- `pca_baseline_results_6class.csv`
- `jitter_actuator_6class.csv`
- `jitter_landmark_6class.csv`

## Reproducibility Tables

- `actuator_definition_table.csv`
- `optimization_hyperparameters.csv`

## Regenerating Summary Figures

From the project root:

```powershell
& 'C:\Users\31734\anaconda3\python.exe' .\generate_submission_figures.py
```

This regenerates:

- `dataset_distribution_6class.png`
- `main_results_overview_6class.png`
- `smoothing_baselines_6class.png`
- `pca_comparison_6class.png`
