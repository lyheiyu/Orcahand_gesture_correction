# Final Compact ORCA Specification

Frozen at: `2026-08-27T12:55:00.777371+00:00`

This specification was produced before final-test evaluation. The final-test labels and performance were not used for actuator ranking, K selection, preprocessing, or classifier settings.

- Development sequences: 456
- Frozen final-test sequences: 115
- Outer split seed: 20260827
- Development repeats: 2
- Few-shot training: 3 sequences per class
- Encoding: Resample-16
- Combined development score: 0.7 Macro-F1 + 0.3 Accuracy, averaged across four classifiers
- K rule: smallest candidate within one standard error of the best development mean
- Chosen K*: **5**
- Encoded features: **80**
- Reduction from ORCA-17: **70.6%**
- Reduction from JointAngle-11: **54.5%**

## Frozen Actuators

| Index | Actuator | Role | Utility |
|---:|---|---|---:|
| 3 | `right_p-pip_actuator` | little PIP flexion | 0.3592 |
| 6 | `right_r-pip_actuator` | ring PIP flexion | 0.8590 |
| 9 | `right_m-pip_actuator` | middle PIP flexion | 0.6765 |
| 12 | `right_i-pip_actuator` | index PIP flexion | 0.7327 |
| 15 | `right_t-mcp_actuator` | thumb MCP flexion | 0.4003 |

## Candidate Development Results

| K | Features | Combined mean | 95% CI | Eligible |
|---:|---:|---:|---:|---|
| 5 | 80 | 0.7960 | 0.0927 | yes |
| 7 | 112 | 0.8173 | 0.0480 | yes |
| 9 | 144 | 0.7929 | 0.0503 | yes |
| 11 | 176 | 0.7639 | 0.0815 | no |
| 13 | 208 | 0.7532 | 0.0997 | no |

## Frozen Parameters

Utility = discriminative - 0.25 within-class - 0.20 instability - 0.15 redundancy - 0.15 refinement-sensitivity. Components are scaled across actuators using development data only.
Semantic constraint: at least one actuator from each of thumb, index, middle, ring, and little; remaining positions are filled by descending utility.

Development manifest SHA256: `b9d3769a00107f29941dc60b58baacec39e113cd30d1d1c1d775546f1c503b17`
Final-test manifest SHA256: `dc995a69e178393bbf1cc056d084185d935b1fcee4248e0cea17cc314711a33f`

The subset must not be changed after inspecting final-test performance.
