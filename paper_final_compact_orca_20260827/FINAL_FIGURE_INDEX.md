# Final Figure Index

All publication figures are in `paper_final_compact_orca_20260827/figures/`.

| Figure | File | Purpose | Main/Supporting |
|---:|---|---|---|
| 1 | `figure_01_final_pipeline.png` | Explain the frozen projection-refinement-compact-readout pipeline | Main |
| 2 | `figure_02_orca_actuator_structure.png` | Show all 17 actuators and the selected seven | Main |
| 3 | `figure_03_representative_trajectory.png` | Visualize a systematically selected trajectory example | Main |
| 4 | `figure_04_temporal_stability.png` | Report actuator-space velocity and acceleration | Main |
| 5 | `figure_05_dimension_control.png` | Compare dimension-controlled representations | Supporting |
| 6 | `figure_06_development_selection.png` | Document development-only K selection | Main |
| 7 | `figure_07_final_compact_vs_jointangle.png` | Show frozen Compact Refined-7 vs JointAngle-11 across classifiers | Main |
| 8 | `figure_08_performance_vs_dimension.png` | Show accuracy-dimension trade-off | Supporting |
| 9 | `figure_09_controlled_perturbation.png` | Show robustness under controlled corruption | Main |
| 10 | `figure_10_main_svm_confusion_comparison.png` | Compare aggregate SVM class-level errors for JointAngle-11 and Compact Refined-7 | Main |

## Figure Rules

- Figure 3 selection metadata is stored in `tables/figure_03_selection_metadata.csv`.
- Significance stars in Figure 7 appear only when Holm-adjusted `p < 0.05`.
- Figure 4 contains actuator-space values only.
- Do not reuse the old four-confusion-matrix figure as the central result. Confusion matrices are supplementary because repeated paired metrics are the stronger evidence.
- Do not reuse old 56-sequence smoothing plots as final dataset evidence.
- The complete KNN, RandomForest, and MLP confusion matrices are in `figures/supplementary_confusion_matrices/`.
- Aggregate CM counts reuse the same 115 final sequences over 20 training selections and are not independent test samples.

## Regeneration

Run from the project root:

```powershell
python .\generate_final_compact_paper_assets.py
```

The script reads frozen result files and does not retrain or retune the method.
