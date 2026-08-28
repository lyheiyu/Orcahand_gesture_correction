# Compact ORCA Selection Explained

## A. Why may 17D contain redundancy?

The 17 ORCA coordinates include wrist, abduction, opposition, and flexion. Some may be correlated or nearly constant for the current six Chinese-dance gestures. Such coordinates can increase a 3-shot classifier's input size without adding proportional class information.

## B. Why may JointAngle-11 benefit from lower dimension?

Resample-16 converts JointAngle-11 into 176 inputs but ORCA-17 into 272. With only 18 training sequences per repeat, the smaller representation can be easier to estimate. The preceding dimension-control experiment also showed that semantic Flex11 was generally stronger than the corresponding ORCA-17 representation.

## C. Why did Flex11 motivate this experiment?

Flex11 removed wrist and abduction coordinates using a predefined semantic rule. Its improvement suggested that some ORCA coordinates contributed limited additional recognition information under this protocol. The current experiment asks whether development data support an even smaller, still interpretable subset.

## D. Why test an even smaller subset?

A compact representation may reduce model input, training variance, storage, and downstream computation. The scientific goal is not to force a higher score, but to determine the smallest development-supported actuator set before touching the final test.

## E. Why development-only selection is necessary

Actuator ranking, K selection, thresholds, and semantic rules are all model-design decisions. They must be made without final-test feedback so the final test remains an unbiased evaluation of the frozen design.

## F. Why final-test selection would be invalid

Trying several subsets on final-test labels and retaining the best would indirectly train on the test set. The reported score would then include selection luck and would overestimate generalization.

## G. Exact frozen split

The outer stratified sequence split uses seed `20260827`: **456 development** sequences and **115 frozen final-test** sequences. Exact IDs are in `development_sequences.csv` and `final_test_sequences.csv`. The manifests are reused if the script runs again.

## H. Exact actuator scoring method

For each actuator, sequence means provide an ANOVA F score for between-class discrimination. Class-conditional variance measures within-class variability. Z-scaled first and second temporal differences measure instability. Maximum absolute Pearson correlation measures redundancy. Refinement sensitivity penalizes loss of normalized discrimination or increased instability from Corrected to Optimized Action.

All five components are computed using development data only and scaled across the 17 actuators. The frozen score is:

```text
utility = discriminative
          - 0.25 * within_class
          - 0.20 * instability
          - 0.15 * redundancy
          - 0.15 * refinement_sensitivity
```

To keep the result anatomically interpretable, each candidate contains at least one coordinate from each finger. Remaining slots follow descending development utility. No arbitrary subset search is performed.

## I. Exact selected K*

The candidate path was `5, 7, 9, 11, 13`. The best development mean occurred at **K=7**. The predefined one-standard-error rule selected the smallest eligible candidate, **K*=7**. This produces **112** Resample-16 classifier inputs.

| K | Encoded inputs | Combined mean | 95% CI | One-SE eligible |
|---:|---:|---:|---:|---|
| 5 | 80 | 0.7955 | 0.0220 | no |
| 7 | 112 | 0.8088 | 0.0250 | yes |
| 9 | 144 | 0.8085 | 0.0227 | yes |
| 11 | 176 | 0.7921 | 0.0232 | no |
| 13 | 208 | 0.7783 | 0.0208 | no |

## J. Retained actuator names and indices

| Index | Actuator | Meaning | Why retained |
|---:|---|---|---|
| 3 | `right_p-pip_actuator` | little PIP flexion | development utility rank 5; contributes to little coverage |
| 6 | `right_r-pip_actuator` | ring PIP flexion | development utility rank 1; contributes to ring coverage |
| 9 | `right_m-pip_actuator` | middle PIP flexion | development utility rank 3; contributes to middle coverage |
| 11 | `right_i-mcp_actuator` | index MCP flexion | development utility rank 6; contributes to index coverage |
| 12 | `right_i-pip_actuator` | index PIP flexion | development utility rank 2; contributes to index coverage |
| 15 | `right_t-mcp_actuator` | thumb MCP flexion | development utility rank 4; contributes to thumb coverage |
| 16 | `right_t-pip_actuator` | thumb IP flexion | development utility rank 7; contributes to thumb coverage |

## K. Why each removed actuator was not retained

These statements apply only to the current dataset and protocol; they do not imply biological uselessness.

| Index | Actuator | Development interpretation |
|---:|---|---|
| 0 | `right_wrist_actuator` | limited additional development discrimination; strongest penalty was within-class variability |
| 1 | `right_p-abd_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 2 | `right_p-mcp_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 4 | `right_r-abd_actuator` | ranked below the frozen K* cutoff; strongest penalty was temporal instability |
| 5 | `right_r-mcp_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 7 | `right_m-abd_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 8 | `right_m-mcp_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 10 | `right_i-abd_actuator` | ranked below the frozen K* cutoff; strongest penalty was temporal instability |
| 13 | `right_t-cmc_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |
| 14 | `right_t-abd_actuator` | ranked below the frozen K* cutoff; strongest penalty was within-class variability |

## L. Resample-16 dimensions

Each selected trajectory is linearly resampled to 16 ordered time positions. Compact OA-7 therefore has `7 x 16 = 112` inputs, compared with `11 x 16 = 176` for JointAngle and `17 x 16 = 272` for full ORCA. Temporal order is retained by flattening the 16 samples in order.

## M. Classifier training

Within each development repeat, ranking uses only the inner training partition. The classifier then receives three sequences per class, while validation sequences remain separate. SVM, KNN, RandomForest, and MLP use the fixed paper hyperparameters. The model pipeline fits StandardScaler on classifier training data only.

## N. Statistics

Development selection averages `0.7 * Macro-F1 + 0.3 * Accuracy` across all four classifiers for each repeat. The one-standard-error rule compares repeat-level combined scores and prefers the smallest K within one standard error of the best mean. Final paired tests will use identical repeat-level training selections and the same frozen final-test sequences; frames will not be treated as independent samples.

## O. How to interpret possible final outcomes

- Compact OA-7 above JointAngle-11 with supported paired statistics: compact refined ORCA wins under this frozen protocol.
- Compact OA-7 similar to JointAngle-11: comparable recognition with 36.4% fewer frame dimensions.
- JointAngle-11 higher: human-joint geometry remains better matched to this classification task.
- Compact OA above OA-Flex11/OA-17: additional coordinates were redundant or noisy for this protocol.
- Compact Corrected above Compact OA: refinement did not improve recognition for the selected coordinates.

## P. Relevant code

- `run_compact_orca_selection.py::freeze_outer_split`: creates or reloads the immutable outer manifests.
- `run_compact_orca_selection.py::actuator_scores`: calculates the five development-only score components.
- `run_compact_orca_selection.py::select_semantic_subset`: enforces five-finger coverage and utility ordering.
- `run_compact_orca_selection.py::development_validation`: reranks inside every inner training partition.
- `run_compact_orca_selection.py::select_k`: applies the four-classifier one-standard-error rule.
- `evaluate_sequence_encodings.py::encode_sequence`: performs Resample-16.
- `train_svm.py::_build_model`: training-only scaling and fixed classifiers.

Core selection snippet:

```python
selected = best_actuator_per_finger(score_rows)
selected += remaining_actuators_in_utility_order
selected = selected[:k]
```

## Q. Plain-language conclusion

Development analysis selected a seven-actuator representation containing flexion coordinates from all five fingers. K=7 slightly exceeded K=9 in the predefined combined development score and reduced ORCA-17 by 58.8%. **The final test has not been evaluated**, so no claim about superiority over JointAngle-11 or OA-Flex11 is made yet. The next permitted action is a single evaluation using the frozen specification.
