# Updated Six-Class Results

## Data provenance

- Source dataset: `gesture_sequence_dataset_chinese_dance_6class.csv`
- Recomputed dataset: `gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`
- Smoothing dataset: `gesture_sequence_dataset_chinese_dance_6class_after_fix_smoothing.csv`
- Split manifest: `split_manifest_6class.csv`
- Figure directory: `figures/paper_updated_6class_20260820/`

The recomputed dataset uses the current palm-normal convention. A validation
sample from six sequences and 18 frames matched a fresh optimizer run exactly,
all generated actuator values remained inside their MuJoCo control bounds, and
all optimized values were finite.

## Dataset

| Gesture | Sequences | Frames |
|---|---:|---:|
| Deer horn | 95 | 4,120 |
| Flower pinch | 94 | 5,013 |
| Orchid finger | 92 | 4,355 |
| Orchid palm | 108 | 4,613 |
| Prayer beads | 92 | 4,290 |
| Three-finger bent | 90 | 3,869 |
| **Total** | **571** | **26,260** |

The CSV has no `subject_id` or `session_id`. The current evidence is therefore
sequence-disjoint but not demonstrably subject-independent or session-independent.

## Evaluation protocol

- 20 deterministic repeats, using seeds 42--61.
- Stratified 20% sequence-level test split in each repeat.
- Three training sequences per class: 18 training sequences in total.
- 115 test sequences in each repeat.
- Identical sequence IDs are used for every representation and classifier.
- StandardScaler and PCA are fitted only on training data.

## Main classification result

Accuracy is reported as mean +/- standard deviation over 20 repeats.

| Classifier | Raw | PCA-17 | Corrected | Optimized Action | Optimized Full |
|---|---:|---:|---:|---:|---:|
| SVM | 0.3887 +/- 0.0380 | 0.6339 +/- 0.0502 | **0.7491 +/- 0.0639** | 0.7278 +/- 0.0536 | 0.4578 +/- 0.0766 |
| KNN | 0.3543 +/- 0.0517 | 0.5822 +/- 0.0636 | **0.7030 +/- 0.0733** | 0.6804 +/- 0.0641 | 0.4152 +/- 0.0638 |
| RandomForest | 0.4309 +/- 0.0442 | 0.6543 +/- 0.0555 | **0.7826 +/- 0.0611** | 0.7674 +/- 0.0626 | 0.5061 +/- 0.0921 |
| MLP | 0.3922 +/- 0.0504 | 0.6191 +/- 0.0507 | **0.7478 +/- 0.0716** | 0.7330 +/- 0.0692 | 0.4574 +/- 0.0765 |

Optimized Action exceeds Raw in all 20 repeats for every classifier. Its mean
accuracy gain over Raw ranges from 0.3261 to 0.3409. However, Corrected exceeds
Optimized Action by 0.0148--0.0226 on average. This difference favors Corrected
significantly for SVM, KNN, and RandomForest in the exploratory paired tests,
but not for MLP.

## Training-size sensitivity (shot sweep)

The primary paper experiment uses **3 training sequences per class**, or 18
training sequences in total. To test whether the conclusion depends on this
choice, the main sensitivity sweep uses 1, 3, 5, and 10 training sequences per
class. Each of the 20 repeats holds out a fixed stratified 20% test set (115
sequences), and the training subsets are nested within that repeat:

`1-shot subset 3-shot subset 5-shot subset 10-shot`.

The sweep uses a newly generated nested split manifest, so its 3-shot values
are a sensitivity result and differ slightly from the primary three-shot table.

### RandomForest accuracy

| Shot per class | Total train | Raw | PCA-17 | Corrected | Optimized Action |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.2878 | 0.3939 | **0.5417** | 0.5296 |
| 3 | 18 | 0.4187 | 0.6687 | **0.7748** | 0.7683 |
| 5 | 30 | 0.5239 | 0.7743 | **0.8561** | 0.8491 |
| 10 | 60 | 0.6813 | 0.8874 | **0.9048** | 0.9013 |

At 1-shot, Corrected and Optimized Action improve over Raw by 0.2539 and 0.2417
accuracy points under RandomForest. At 10-shot they reach 0.9048 and 0.9013,
compared with 0.6813 for Raw and 0.8874 for PCA-17. The structured actuator
representations therefore retain their advantage throughout the intended
low-data regime. Corrected remains slightly more discriminative, while
Optimized Action provides the substantially smoother latent trajectory.

The earlier 1--40-shot exploratory sweep is retained in
`shot_sweep_1to40_archive/` for transparency, but it is not used as the main
few-shot figure because 20 and 40 training sequences per class no longer
represent the paper's strict low-shot setting.

The RandomForest confusion matrices show that flower pinch, orchid finger, and
orchid palm account for many of the errors in the 1-shot condition. Most class
recalls approach or exceed 0.85 by 10 shots. Orchid palm remains the most difficult
10-shot class for Optimized Action, suggesting that future data collection
should prioritize within-class variation for this gesture.

## Temporal stability

Actuator-space values must be interpreted separately from landmark-space values.

| Actuator representation | Velocity mean | Acceleration mean |
|---|---:|---:|
| Corrected | 0.4590 | 0.7100 |
| Optimized Action | **0.2470** | **0.2703** |

Relative to Corrected, Optimized Action reduces mean actuator velocity by 46.2%
and mean actuator acceleration by 61.9%.

| Landmark representation | Velocity mean | Acceleration mean |
|---|---:|---:|
| Raw | 0.6043 | 0.7906 |
| Moving Average | 0.3340 | 0.1907 |
| Savitzky-Golay | 0.4391 | 0.2951 |
| One-Euro | 0.3665 | 0.2857 |
| Kalman | **0.1737** | **0.0974** |
| Optimized Full | 0.3011 | 0.3127 |

Kalman is the strongest coordinate smoother, but it does not provide the best
recognition representation.

## Smoothing comparison

Under RandomForest, Raw reaches 0.4309 accuracy. Moving Average, Savitzky-Golay,
One-Euro, and Kalman reach 0.4248, 0.4204, 0.4187, and 0.4096, respectively.
Corrected and Optimized Action reach 0.7826 and 0.7674. Thus, lower coordinate
jitter does not imply better gesture recognition, and ordinary smoothing does
not explain the actuator-space advantage.

## Runtime

- 300-frame solver-only benchmark.
- Mean: 27.61 ms/frame.
- Median: 27.27 ms/frame.
- 95th percentile: 32.62 ms/frame.
- Mean L-BFGS-B iterations: 6.02.
- Success and finite-output rates: 100%.

The timing excludes MediaPipe inference, capture, rendering, and classification;
the system should therefore be described as causal frame-wise processing rather
than as proven end-to-end real time.

## Loss ablation

| Variant | Velocity mean | Acceleration mean | SVM accuracy | RF accuracy |
|---|---:|---:|---:|---:|
| Corrected only | 0.4590 | 0.7100 | **0.7491** | **0.7826** |
| Full | 0.2470 | 0.2703 | 0.7278 | 0.7674 |
| Without palm normal | **0.2391** | **0.2646** | 0.7378 | 0.7717 |
| Without acceleration | 0.2685 | 0.3718 | 0.7213 | 0.7730 |
| Without temporal terms | 0.3238 | 0.4976 | 0.7096 | 0.7609 |
| L2 instead of Huber | 0.2560 | 0.2926 | 0.6548 | 0.7335 |

The temporal terms markedly reduce actuator variation. Full also exceeds the
no-temporal variant under SVM, although the RandomForest difference is small.
Huber fitting is the clearest discriminative component: replacing it with L2
reduces SVM accuracy by 0.0730 and RandomForest accuracy by 0.0339, while raising
mean solve time to 74.9 ms/frame. Palm-normal alignment remains a kinematic
consistency term but does not improve classification in this ablation.

## Revised paper conclusion

The enlarged dataset supports two distinct claims. First, semantic actuator-space
representations outperform raw landmarks, conventional smoothing, and PCA-17
under all four lightweight classifiers. Second, MuJoCo-constrained temporal
optimization substantially stabilizes the actuator trajectory while preserving
most, but not all, of the discriminative information in the heuristic Corrected
representation. The main finding is therefore a stability--discrimination
trade-off, not universal classification dominance by Optimized Action.

## Reproduction

The exact numerical outputs and figures are retained in the paths listed above.
Do not replace these sequence-disjoint results with cross-subject claims until
participant and session identifiers are collected.
