# Updated six-class paper figures

All figures and CSV summaries in this directory were regenerated from the
current 571-sequence Chinese dance dataset after recomputing Optimized Action
and Optimized Full with the corrected palm-normal convention.

## Main-text figures

- `method_pipeline.png`: method and evaluation pipeline.
- `dataset_distribution_6class.png`: per-class sequence and frame counts.
- `jitter_actuator_6class.png`: Corrected versus Optimized Action in actuator space.
- `jitter_landmark_6class.png`: Raw, smoothing baselines, and Optimized Full in landmark space.
- `trajectory_refinement_example_6class.png`: representative high-jitter actuator trajectory.
- `paired_accuracy_improvement_6class.png`: paired Optimized Action differences on identical splits.
- `smoothing_baselines_6class.png`: RandomForest recognition after conventional smoothing.
- `performance_heatmaps_6class.png`: four-classifier representation comparison.
- `cm_svm_representation_stages_6class.png`: Raw, Corrected, and Optimized Action SVM confusion matrices.
- `pca_all_classifiers_6class.png`: dimension-matched PCA-17 comparison.
- `loss_ablation_stability_tradeoff_6class.png`: loss-component stability/recognition analysis.
- `shot_sweep_accuracy_6class.png`: accuracy learning curves for 1, 3, 5, and 10 training sequences per class.
- `shot_sweep_macro_f1_6class.png`: macro-F1 learning curves under the same nested low-shot protocol.

## Appendix figures

- `per_class_recall_svm_6class.png`: class-wise SVM recall.
- `cm_optimized_action_all_classifiers_6class.png`: Optimized Action confusion matrices.
- `loss_ablation_6class.png`: classification-only loss ablation.
- `implementation_validation_before_after.png`: palm-normal regression and synthetic recovery check.
- `shot_sweep_optimized_minus_corrected_6class.png`: paired recognition difference between temporal refinement and its heuristic initialization.
- `cm_shot_progression_rf_optimized_action_6class.png`: RandomForest Optimized Action confusion matrices from 1-shot to 10-shot.
- `cm_low_vs_high_shot_rf_representations_6class.png`: Raw, PCA-17, Corrected, and Optimized Action confusion matrices at 1, 3, 5, and 10 shots.
- `shot_sweep_per_class_recall_rf_6class.png`: class-wise recall as the training set grows.

## Reproducibility files

The CSV files contain the numerical values used to create the plots. The exact
split manifest and regenerated datasets are stored in
`diagnostics/updated_6class_20260820/`.

The shot-sweep CSV files are stored in this directory. The sweep holds out the
same stratified 20% test partition within each repeat and uses nested training
sets, so that every smaller-shot training set is a subset of the larger-shot
sets. The primary paper protocol remains the separately saved three-shot
experiment; the sweep is a training-size sensitivity analysis.

Regenerate the complete shot sweep with:

```powershell
python .\generate_shot_sweep_figures.py `
  --dataset .\diagnostics\updated_6class_20260820\gesture_sequence_dataset_chinese_dance_6class_after_fix.csv `
  --output-dir .\figures\paper_updated_6class_20260820 `
  --shots 1 3 5 10 `
  --repeats 20
```

Do not combine actuator-space and landmark-space jitter values in one ranking;
the two spaces have different dimensions and scales.
