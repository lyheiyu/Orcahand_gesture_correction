# Final Compact ORCA Specification

Frozen at: `2026-08-27T12:58:26.381966+00:00`

This specification was produced before final-test evaluation. The final-test labels and performance were not used for actuator ranking, K selection, preprocessing, or classifier settings.

- Development sequences: 456
- Frozen final-test sequences: 115
- Outer split seed: 20260827
- Development repeats: 20
- Few-shot training: 3 sequences per class
- Encoding: Resample-16
- Combined development score: 0.7 Macro-F1 + 0.3 Accuracy, averaged across four classifiers
- K rule: smallest candidate within one standard error of the best development mean
- Chosen K*: **7**
- Encoded features: **112**
- Reduction from ORCA-17: **58.8%**
- Reduction from JointAngle-11: **36.4%**

## Frozen Actuators

| Index | Actuator | Role | Utility |
|---:|---|---|---:|
| 3 | `right_p-pip_actuator` | little PIP flexion | 0.3592 |
| 6 | `right_r-pip_actuator` | ring PIP flexion | 0.8590 |
| 9 | `right_m-pip_actuator` | middle PIP flexion | 0.6765 |
| 11 | `right_i-mcp_actuator` | index MCP flexion | 0.2338 |
| 12 | `right_i-pip_actuator` | index PIP flexion | 0.7327 |
| 15 | `right_t-mcp_actuator` | thumb MCP flexion | 0.4003 |
| 16 | `right_t-pip_actuator` | thumb IP flexion | 0.1957 |

## Candidate Development Results

| K | Features | Combined mean | 95% CI | Eligible |
|---:|---:|---:|---:|---|
| 5 | 80 | 0.7955 | 0.0220 | no |
| 7 | 112 | 0.8088 | 0.0250 | yes |
| 9 | 144 | 0.8085 | 0.0227 | yes |
| 11 | 176 | 0.7921 | 0.0232 | no |
| 13 | 208 | 0.7783 | 0.0208 | no |

## Frozen Parameters

Utility = discriminative - 0.25 within-class - 0.20 instability - 0.15 redundancy - 0.15 refinement-sensitivity. Components are scaled across actuators using development data only.
Semantic constraint: at least one actuator from each of thumb, index, middle, ring, and little; remaining positions are filled by descending utility.

Development manifest SHA256: `b9d3769a00107f29941dc60b58baacec39e113cd30d1d1c1d775546f1c503b17`
Final-test manifest SHA256: `dc995a69e178393bbf1cc056d084185d935b1fcee4248e0cea17cc314711a33f`

The subset must not be changed after inspecting final-test performance.
