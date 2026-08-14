# Paper Rewrite Main Artifacts

This folder is the self-contained figure and result-artifact folder for `main.tex`.

## Current Result Version

The main-paper artifacts were regenerated on 2026-08-14 after correcting the palm-normal sign convention and completing the PCA, ablation, and runtime experiments. The current optimized features come from:

- `diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`
- fixed protocol: 20 repeats, three shots per class, sequence-level split, random state 42

The authoritative structured-classification table is `classification_suite_6class.csv`. The compact table consumed by the paper figure generator is `main_results_6class.csv`.

## Figures Used In The Main Draft

- `method_pipeline.png`
- `dataset_distribution_6class.png`
- `jitter_actuator_6class.png`
- `jitter_landmark_6class.png`
- `trajectory_refinement_example_6class.png`
- `paired_accuracy_improvement_6class.png`
- `smoothing_baselines_6class.png`
- `performance_heatmaps_6class.png`
- `cm_svm_representation_stages_6class.png`
- `loss_ablation_stability_tradeoff_6class.png`
- `pca_all_classifiers_6class.png`

## Appendix Figures

- `per_class_recall_svm_6class.png`
- `cm_optimized_action_all_classifiers_6class.png`
- `loss_ablation_6class.png`
- `implementation_validation_before_after.png`

## Main Result CSVs

- `dataset_summary_6class.csv`
- `main_results_6class.csv`
- `smoothing_baseline_results_6class.csv`
- `pca_baseline_results_6class.csv`
- `jitter_actuator_6class.csv`
- `jitter_landmark_6class.csv`
- `per_repeat_scores_6class.csv`
- `paired_accuracy_stats_6class.csv`
- `per_class_recall_svm_6class.csv`
- `trajectory_refinement_example_6class.csv`
- `pca_all_classifiers_per_repeat_6class.csv`
- `pca_all_classifiers_summary_6class.csv`
- `loss_ablation_per_repeat_6class.csv`
- `loss_ablation_summary_6class.csv`
- `loss_ablation_paired_stats_6class.csv`
- `loss_ablation_stability_6class.csv`

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

The classifier plots and aggregate confusion matrices are regenerated with:

```powershell
& 'C:\Users\31734\anaconda3\envs\orca\python.exe' .\generate_classifier_figures.py `
  --dataset .\diagnostics\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --output-dir .\figures\paper_rewrite_main `
  --results-csv classification_suite_6class.csv `
  --sequence-mode --shots-per-class 3 --repeats 20 --test-size 0.2 --random-state 42 `
  --classifiers svm knn rf mlp `
  --feature-sets raw corrected optimized_action optimized_full
```

Files with older generic names such as `representation_comparison.png`, `best_cm_*.png`, and `smoothing_comparison.png` are retained for history but are not referenced by `main.tex`.

The extended evidence figures are regenerated with:

```powershell
& 'C:\Users\31734\anaconda3\envs\orca\python.exe' .\generate_extended_paper_figures.py
```

This command re-evaluates every representation on the saved split manifest before drawing paired comparisons, heatmaps, class-level recall, confusion panels, and the trajectory diagnostic.

The completed PCA-17, loss-ablation, and runtime experiments are reproduced with:

```powershell
& 'C:\Users\31734\anaconda3\envs\orca\python.exe' .\run_paper_completion_experiments.py `
  --dataset .\diagnostics\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --split-manifest .\diagnostics\palm_fix_split_manifest_6class.csv `
  --workers 4 --runtime-max-frames 300
```

The expensive per-frame ablation trajectories are cached under `diagnostics/paper_completion`. Re-running the command reuses them unless `--force-ablation` is supplied.
