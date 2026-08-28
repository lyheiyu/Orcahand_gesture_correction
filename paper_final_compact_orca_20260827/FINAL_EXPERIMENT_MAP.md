# Final Experiment Map

| Research question | Experiment | Frozen source | Main output | Supported conclusion |
|---|---|---|---|---|
| RQ1 structured representation | 17D actuator and dimension-controlled comparison | `final_test_results.csv` | Fig. 5, Table 4 | ORCA projection is competitive and interpretable |
| RQ2 temporal stability | Actuator Projection-17 vs Refined ORCA-17 on 571 sequences | `UPDATED_RESULTS.md` | Figs. 3-4, Table 6 | Refinement reduces actuator velocity and acceleration |
| RQ2 perturbation robustness | Gaussian, spike, dropout benchmark | `actuator_overall_summary.csv`, `actuator_summary_by_corruption.csv` | Fig. 9, Table 7 | Refinement reduces actuator sensitivity to corruption |
| RQ3 compactness | Development-only K scan and semantic coverage | `compact_dimension_selection.csv` | Figs. 2 and 6, Tables 2-3 | K=7 was frozen without final-test labels |
| RQ4 external baseline | Compact Refined-7 vs JointAngle-11, four classifiers | `final_test_results.csv` | Fig. 7, Table 4 | Compact Refined-7 is higher for SVM/KNN; classifier-dependent otherwise |
| RQ4 significance | Repeat-wise paired tests and Holm correction | `final_test_paired_comparisons_holm.csv` | Table 5 | SVM/KNN significant; RF/MLP not significant |
| Dimension control | Flex11 and PCA11 supporting comparisons | `final_test_results.csv` | Figs. 5 and 8 | Benefits are not explained by dimension alone |
| Objective components | Frozen no-palm/no-acceleration/no-temporal/L2 ablations | `loss_ablation_summary_6class.csv` | Table 8 | Temporal and robust terms matter; palm effect is smaller |
| Feasibility | Runtime, finite output, optimizer success | `runtime_summary_6class.json` | Table 9 | Causal implementation converges on diagnostic sample |

## Evidence Separation Rules

- Development evidence selects Compact7; it must not be reported as final-test performance.
- Final-test evidence compares frozen representations; it must not be used to redesign Compact7.
- Stability evidence compares only common actuator-space quantities.
- Perturbation evidence measures sensitivity, not human pose error.
- Older 39- and 56-sequence experiments are historical and excluded from the final story.
