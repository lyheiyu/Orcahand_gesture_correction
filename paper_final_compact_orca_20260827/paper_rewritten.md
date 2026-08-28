# Compact ORCA Actuator Representations with MuJoCo-Constrained Temporal Refinement for Few-Shot Chinese Dance Gesture Recognition

Jiating Li

## Abstract

Monocular hand landmarks provide an accessible input for gesture recognition, but their high dimensionality and temporal instability are problematic when training data are limited. This paper presents a structured representation pipeline for Chinese dance hand gestures. MediaPipe landmarks are first mapped frame by frame to a 17-dimensional actuator state of an ORCA articulated hand. The state is then refined causally using MuJoCo forward kinematics, robust observation fitting, actuator bounds, and first- and second-order temporal regularization. A development-only selection procedure further freezes a seven-actuator semantic subset, Compact Refined-7, before final-test evaluation. Experiments use 571 sequences from six gesture classes. Within actuator space, temporal refinement reduces mean velocity from 0.4590 to 0.2470 and mean acceleration from 0.7100 to 0.2703, corresponding to reductions of 46.2% and 61.9%. Under controlled landmark corruption, refinement reduces the overall actuator deviation from 0.0292 to 0.0182, a 37.6% reduction relative to Actuator Projection-17. On the frozen 115-sequence final test, Compact Refined-7 uses 112 sequence features and outperforms the published-definition JointAngle-11 baseline with SVM and KNN after Holm correction, while the RandomForest and MLP differences are not significant. These findings support a bounded claim: ORCA embodiment provides a compact semantic state, MuJoCo-constrained temporal refinement improves its stability and perturbation robustness, and a frozen semantic subset can preserve or improve recognition performance without claiming ground-truth 3D pose recovery.

**Keywords:** hand gesture recognition; MediaPipe; MuJoCo; actuator representation; temporal refinement; few-shot learning; Chinese dance

## 1. Introduction

Hand gestures in Chinese dance are defined by coordinated finger flexion, thumb articulation, and palm configuration. Vision-based capture is attractive because it avoids gloves or markers, but practical monocular trackers can produce jitter, transient spikes, and unstable joint configurations under oblique viewpoints, rapid movement, and self-occlusion. These errors matter in few-shot recognition because a classifier must separate subtle gesture classes using only a small number of labeled sequences.

MediaPipe Hands provides an efficient landmark detector and lightweight tracking pipeline (Zhang et al., 2020). Existing gesture-recognition systems commonly treat estimated landmarks as ready-to-use features and focus on the temporal classifier. This is effective when observations are reliable, but it leaves two representation problems unresolved. First, 21 three-dimensional landmarks form a 63-dimensional coordinate vector whose components include nuisance variation. Second, coordinate-space smoothing can reduce high-frequency variation without enforcing compatibility with an articulated hand model.

This work asks whether an embodied actuator space can provide a more compact and stable intermediate representation. The proposed pipeline does not introduce a new deep sequence classifier. Instead, it maps landmarks to the 17 actuators of an ORCA hand, refines the actuator trajectory through MuJoCo forward kinematics and causal temporal regularization, and evaluates the resulting representation with lightweight classifiers. This distinction is important: classification is used to evaluate representation utility, whereas the main method operates before classification.

The study addresses four research questions:

- **RQ1:** How does the 17-dimensional actuator projection reorganize MediaPipe landmarks into a bounded semantic state, and how does it compare with other structured representations under dimension-controlled evaluation?
- **RQ2:** Does MuJoCo-constrained causal refinement reduce actuator-space temporal variation and sensitivity to controlled landmark corruption?
- **RQ3:** Can a development-selected semantic actuator subset retain useful information with fewer dimensions?
- **RQ4:** On a frozen final test, how does Compact Refined-7 compare with a conventional 11-dimensional 3D joint-angle representation across classifiers?

The contributions are:

1. An embodiment-aware frame-wise mapping from MediaPipe landmarks to a 17-dimensional ORCA actuator state, providing an interpretable alternative to raw coordinates and generic dimensionality reduction.
2. A causal MuJoCo-constrained refinement objective combining robust landmark fitting, palm orientation, actuator priors, hard bounds, and temporal regularization. Its main demonstrated benefit is reduced temporal variation and reduced actuator sensitivity under controlled input corruption.
3. A development-only compact-actuator selection procedure that freezes seven semantically distributed flexion actuators before final evaluation. Compact Refined-7 reduces encoded dimensionality by 58.8% relative to Refined ORCA-17 and by 36.4% relative to JointAngle-11.
4. A dimension-controlled and classifier-controlled evaluation using a frozen 456/115 development/final split, common repeated few-shot splits, confidence intervals, paired Wilcoxon tests, Holm correction, and effect sizes.

The paper deliberately avoids three stronger claims that the current evidence cannot support. MuJoCo output is not treated as ground-truth human 3D pose; controlled corruption is not called physical pose recovery; and the refinement is not claimed to improve every classifier or every representation.

## 2. Related Work

### 2.1 Vision-based hand landmarks and gesture recognition

MediaPipe Hands combines palm detection with hand landmark estimation for real-time use (Zhang et al., 2020). Markerless methods are convenient, but monocular hand pose remains ambiguous because fingers self-occlude and multiple 3D configurations can produce similar projections (Zimmermann and Brox, 2017; Boukhayma et al., 2019). Gesture datasets and recognition systems have consequently explored RGB, skeleton, and landmark representations (Caputo et al., 2021; Kopuklu et al., 2020; Kapitanov et al., 2022).

### 2.2 Structured hand representations

Raw landmarks are not the only representation of hand articulation. Joint angles describe local geometry, while articulated hand models restrict motion to meaningful degrees of freedom. MANO demonstrates the value of a compact hand parameterization (Romero et al., 2017). The present study uses a different embodiment: the actuator space of an ORCA robotic hand. This is not asserted to be anatomically identical to a human hand. It is used as an interpretable, bounded, low-dimensional state space.

### 2.3 Temporal filtering and model-constrained refinement

Moving averages, Savitzky-Golay filtering (Savitzky and Golay, 1964), the One Euro filter (Casiez et al., 2012), and Kalman filtering (Kalman, 1960) suppress temporal noise directly in coordinate space. They are important baselines for landmark smoothness, but they do not impose the ORCA kinematic structure. MuJoCo provides a forward model for articulated systems (Todorov et al., 2012). Our use of MuJoCo is kinematic and optimization based: actuator states are evaluated through forward kinematics, not through learned dynamics, forces, or contact recovery.

### 2.4 Few-shot evaluation

Few-shot learning emphasizes representation quality when labeled examples are scarce (Vinyals et al., 2016; Snell et al., 2017). Rather than introducing a meta-learning model, this paper uses repeated fixed-shot evaluations with lightweight classifiers. This intentionally limits classifier capacity so that representation differences remain visible.

## 3. Method

### 3.1 Overview

For frame `t`, let `Y_t in R^(21x3)` denote normalized MediaPipe landmarks and let `q_t in R^17` denote the ORCA actuator state. The pipeline is

```text
MediaPipe landmarks Y_t
  -> Actuator Projection-17 q_tilde
  -> causal MuJoCo refinement q_star (Refined ORCA-17)
  -> frozen Compact Refined-7 readout
  -> Resample16 sequence encoding
  -> lightweight classifier
```

The first stage supplies a structured initialization. The second stage addresses temporal consistency. The compact readout addresses efficiency and dimension control. These roles must not be conflated.

### 3.2 Actuator Projection-17

The frame-wise actuator projection is

`q_tilde_t = g(Y_t), q_tilde_t in R^17`,

where `g` is a deterministic geometric mapping. Landmarks are normalized at the wrist and converted into palm orientation, finger abduction, finger flexion, and thumb articulation features. Each feature is mapped to the range of its corresponding ORCA actuator. The full actuator inventory is reported in Table 1.

This projection is stored under the legacy feature key `corrected` in the code. The paper uses **Actuator Projection-17** consistently because it is a deterministic projection rather than an optimization output. It can nevertheless be highly discriminative: the geometry-to-actuator mapping removes nuisance coordinates and preserves gesture-relevant articulation.

The mapping follows the implementation. For landmarks `a`, `b`, and `c`, the included joint angle is

`theta(a,b,c) = acos( ((a-b) dot (c-b)) / (||a-b|| ||c-b||) )`.

Finger flexion is normalized as `f(theta)=clip((175-theta_deg)/95,0,1)`. Signed finger spread is measured around the palm normal with

`phi(v1,v2,n)=atan2((v1 x v2) dot n, v1 dot v2)`

and normalized as `clip(phi_deg/25,-1,1)`. The palm basis uses wrist-to-middle-MCP as the forward vector, pinky-MCP-to-index-MCP as the across vector, and `unit(forward x across)` as the normal. Wrist flexion is `clip(atan2(-forward_z, |forward_y|)/55 deg,-1,1)`. Thumb opening combines a signed planar opening term and a normalized thumb-tip-to-index-MCP distance with weights 0.7 and 0.3. Unit features are mapped linearly to `[q_min,q_max]`; signed features are mapped about either the range midpoint or the neutral MuJoCo state. Table 1 reports all actuator bounds in radians. These fixed formulas explain why the projection is an embodiment-aware reparameterization rather than PCA or coordinate smoothing.

### 3.3 MuJoCo-constrained causal temporal refinement

Let `h(q_t)` denote the MuJoCo forward-kinematic mapping from actuator state to selected model points. The refined state solves

`q_t* = argmin_(q_t in Q) L(q_t)`,

with

`L = lambda_l L_huber + lambda_n L_normal + lambda_p L_prior + lambda_s L_temporal + lambda_a L_acceleration + lambda_d L_default + lambda_b L_boundary`.

The feasible set `Q` is defined by the MuJoCo actuator box limits and enforced by SciPy L-BFGS-B. The implementation initializes every frame from the clipped Actuator Projection-17 state and permits at most 120 iterations; stopping tolerances other than this iteration cap use SciPy defaults. The fixed values are `lambda_l=1.00`, `lambda_n=0.20`, `lambda_p=0.30`, `lambda_s=0.10`, `lambda_a=0.15`, `lambda_d=0.15`, `lambda_b=0.05`, and `delta=0.08`. No weight is tuned per sequence or test condition.

#### Robust observation term

Eight correspondences are used: wrist, thumb tip, index MCP and tip, middle MCP and tip, and little MCP and tip. For coordinate residual `r_(t,i,k) = h_i(q_t)_k - Y_(t,i,k)`,

`L_huber = sum_(i in C) sum_(k=1)^3 rho_delta(r_(t,i,k))`,

where

`rho_delta(r) = 0.5 r^2` if `|r| <= delta`, and

`rho_delta(r) = delta(|r| - 0.5 delta)` otherwise.

The component-wise Huber penalty limits the influence of isolated large coordinate residuals without discarding all observations from the affected frame.

#### Palm-normal term

`L_normal = ||n_orca(q_t) - n_mp(Y_t)||_2^2`.

Both normals use the same cross-product convention. A regression test with MuJoCo-generated observations confirms that their dot product is close to `+1`, preventing the earlier sign inconsistency.

#### Initialization prior

`L_prior = ||q_t - q_tilde_t||_2^2`.

This term anchors optimization to the semantic frame-wise projection. It is important because monocular observations do not uniquely identify every actuator.

#### Temporal terms

`L_temporal = ||q_t - q_(t-1)*||_2^2`,

`L_acceleration = ||q_t - 2q_(t-1)* + q_(t-2)*||_2^2`.

The first term discourages abrupt velocity; the second discourages abrupt changes in velocity. The method is causal because only the current observation and previous optimized states are used. At the first frame both temporal terms are zero. At the second frame only the first-order term is active. State history is reset at every sequence boundary.

#### Default-pose and boundary terms

Let `q^0` be the neutral MuJoCo actuator state:

`L_default = ||q_t - q^0||_2^2`.

For actuator `j`, define the normalized range coordinate

`u_(t,j) = (q_(t,j) - q_j^min) / max(q_j^max - q_j^min, 10^-6)`.

The boundary penalty is

`L_boundary = sum_j [max(0, 0.1-u_(t,j))^2 + max(0, u_(t,j)-0.9)^2]`.

This soft term discourages persistent solutions in the outer 10% of each range; hard L-BFGS-B limits still guarantee feasibility.

### 3.4 Coordinate alignment and reconstructed landmarks

Before point comparison, MediaPipe and MuJoCo points are translated to their wrist origin. Scale is normalized using

`s_t = max(||Y_indexMCP - Y_pinkyMCP||_2, ||Y_middleMCP - Y_wrist||_2, 10^-6)`.

The implementation preserves the acquisition coordinate axes rather than rotating every frame into a new palm-local frame. Experiments use right-hand observations and the right-hand ORCA model. The scope is therefore model-consistent refinement under this acquisition convention, not general left/right canonicalization.

For landmark-space inspection,

`Y_t* = h(q_t*)`

is exported under the legacy code key `optimized_full` and is called **Reconstructed ORCA Landmarks** in the paper. This reconstruction is useful for visualization but is not assumed to be superior for recognition. It returns the low-dimensional state to a higher-dimensional model-point representation and may inherit embodiment mismatch.

### 3.5 Frozen Compact Refined-7

Compact-actuator selection was performed using development data only. Candidate subsets were scored using discriminative utility, temporal stability, redundancy, and semantic finger coverage. The selection score was

`U_j = D_j - 0.25 W_j - 0.20 I_j - 0.15 R_j - 0.15 S_j`,

where `D_j` is the min-max-scaled `log(1+F_j)` from a sequence-level ANOVA F statistic; `W_j` is min-max-scaled within-class variance; `I_j` is min-max-scaled first- plus second-difference RMS; `R_j` is the maximum absolute Pearson correlation with another actuator; and `S_j` penalizes loss of discriminative utility or increased instability after refinement. All components are computed after development-partition standardization. Candidate values `K in {5,7,9,11,13}` are evaluated with a combined development score `0.3 Accuracy + 0.7 Macro-F1`, averaged over the four classifiers. The smallest K whose mean lies within one standard error of the best candidate is selected, subject to at least one channel for each thumb/finger group. Figure 6 displays 95% confidence intervals (`1.96 s/sqrt(20)`), not standard deviations.

The frozen indices are

`[3, 6, 9, 11, 12, 15, 16]`,

corresponding to little PIP, ring PIP, middle PIP, index MCP, index PIP, thumb MCP, and thumb IP flexion. No final-test labels were used to select these indices. Applied to `q_tilde_t`, the subset is **Compact Projection-7**; applied to `q_t*`, it is **Compact Refined-7**.

### 3.6 Sequence encoding

The primary encoder preserves temporal order through linear resampling. For a variable-length sequence `X in R^(Txd)`, each feature trajectory is interpolated to 16 normalized time positions:

`Z = Resample16(X) in R^(16d)`.

Thus Refined ORCA-17 produces 272 features, JointAngle-11 produces 176, and Compact Refined-7 produces 112. The older mean/std/max/delta descriptor is retained only as an order-light supporting baseline; it is not the primary paper result.

### 3.7 External joint-angle baseline

JointAngle-11 computes conventional 3D angles from landmark vectors following the published geometric definition used by the referenced baseline. It is a visible external representation baseline, not a claim of full reproduction of that paper's data, preprocessing, classifier, or experimental protocol.

## 4. Experimental Protocol

### 4.1 Dataset and frozen split

The current dataset contains 571 sequences and 26,260 frames from six Chinese dance gesture classes: deer horn, flower pinch, orchid finger, orchid palm, prayer beads, and three-finger bent. A single frozen split assigns 456 sequences to development and 115 to final testing. Compact-actuator selection and all selection decisions use only the development partition. The final partition is opened once for the frozen comparison.

The submission version must replace the following fields with acquisition records rather than estimates: **[TODO: number of participants]**, **[TODO: sessions per participant and recording dates]**, **[TODO: sequences per class]**, **[TODO: camera/device]**, **[TODO: resolution and frame rate]**, **[TODO: camera distance, lighting, and background protocol]**, and **[TODO: whether failed MediaPipe frames were retained or excluded]**. These fields cannot be recovered reliably from the CSV alone.

The present dataset is not yet a subject-independent benchmark. Participant and session diversity therefore remain central limitations, and conclusions are restricted to the current acquisition domain.

### 4.2 Few-shot evaluation

The primary evaluation uses three labeled training sequences per class and 20 repeated common splits. The same sampled sequence identifiers are used for every representation within a repeat. Splitting and sampling occur at sequence level, never at frame level. Standardization is fitted on training data only. PCA baselines, when used, are also fitted on training data only. Frozen classifier settings are: RBF-SVM (`C=5`, `gamma=scale`); distance-weighted KNN (`k=3`); RandomForest (200 trees, unrestricted depth); and MLP (hidden layers 128 and 64, `alpha=10^-4`, initial learning rate `10^-3`, maximum 1200 iterations). All four estimators are wrapped in a training-only `StandardScaler` pipeline.

We report accuracy, macro-F1, and Cohen's kappa as mean, standard deviation, and 95% confidence interval. Paired representation differences use the repeat-wise Wilcoxon signed-rank test and Cohen's `d_z`. Holm correction is applied separately for each classifier-metric pair over six predefined representation comparisons; thus each correction family contains six hypotheses. Because all 20 repeats evaluate the same frozen 115-sequence final partition, the intervals quantify variation due to few-shot training-sequence selection, not uncertainty for new participants or a new dataset.

### 4.3 Temporal stability

Actuator-space stability is evaluated only between representations in the same 17-dimensional actuator space. Mean first differences measure velocity, and mean second differences measure acceleration. Raw or reconstructed landmark-space results are not numerically compared with actuator-space values.

### 4.4 Controlled perturbation sensitivity

Gaussian noise, isolated spikes, and landmark dropout are added in the normalized raw-landmark coordinate system. Gaussian noise affects every coordinate of all 21 landmarks with standard deviations `0.01`, `0.03`, and `0.06`. A spike selects one distal finger group, adds a fixed-length random-direction displacement of magnitude `0.75`, and lasts one, two, or three frames. Dropout selects one distal finger group and freezes its three distal landmarks at the last visible frame for three or five frames. Conditions are balanced across the 571 sequences, use ten deterministic seed values beginning at 42, and affect 127,704 frame-landmark pairs in total.

Clean and corrupted trajectories are independently projected or refined, then normalized by the same actuator bounds. For a sequence with `T` frames and `d=17` actuators, sensitivity is `MAE = (1/(Td)) sum_(t,j) |q_corrupt_norm(t,j)-q_clean_norm(t,j)|`; the reported value is the mean of this sequence-level MAE across sequences. Dropout means freezing, not zero filling or deleting observations. This experiment tests robustness to controlled input perturbations; it is not a ground-truth human pose recovery benchmark.

### 4.5 Ablation and runtime

Supporting ablations remove palm-normal alignment, acceleration, or all temporal terms, and replace Huber fitting with L2. Runtime is measured on 300 frames using the existing implementation and hardware configuration.

## 5. Results

### 5.1 Temporal refinement

Within the common 17-dimensional actuator space, Actuator Projection-17 has mean velocity 0.4590 and acceleration 0.7100. Refined ORCA-17 reduces these values to 0.2470 and 0.2703, respectively. The relative reductions are 46.2% for velocity and 61.9% for acceleration. Figure 3 shows a systematically selected median-positive example rather than a hand-picked best case, and Figure 4 reports the full-dataset aggregate.

These results answer RQ2 for temporal stability. They do not establish that the refined state is closer to unknown human 3D ground truth.

### 5.2 Controlled corruption

Across controlled corruptions, actuator deviation decreases from 0.0292 for Actuator Projection-17 to 0.0182 for Refined ORCA-17, a 37.6% reduction. The reductions are 40.7% under Gaussian noise, 17.7% for isolated spikes, and 21.9% for dropout. This supports robustness of the latent actuator trajectory to observation perturbations.

Conventional Kalman and One Euro filters can provide stronger direct smoothing in landmark space. This does not contradict the present result: their objective is coordinate smoothing, whereas ORCA refinement targets a bounded articulated latent state. Reconstructed ORCA Landmarks may also show a clean-domain reconstruction bias because the ORCA geometry is not ground-truth human anatomy.

### 5.3 Compactness and frozen final classification

Development-only selection chose seven actuators. With Resample16, Compact Refined-7 uses 112 features, compared with 272 for Refined ORCA-17 and 176 for JointAngle-11. These are reductions of 58.8% and 36.4%, respectively.

On the frozen final test, Compact Refined-7 reaches accuracy/macro-F1 of 0.8400/0.8438 with SVM, 0.8070/0.8091 with KNN, 0.8235/0.8222 with RandomForest, and 0.8143/0.8107 with MLP. Compared with JointAngle-11, the repeat-wise accuracy differences are +0.0374 for SVM, +0.0448 for KNN, -0.0087 for RandomForest, and +0.0183 for MLP. After Holm correction, the SVM and KNN improvements are significant; the RandomForest and MLP differences are not.

The classifier-dependent result is scientifically useful. Compact Refined-7 is not universally best, but it provides a smaller semantic representation that is clearly beneficial for distance- and margin-based classifiers in the frozen evaluation.

Figure 10 compares the aggregate SVM confusion matrices for JointAngle-11 and Compact Refined-7. Each cell reports row-normalized frequency and cumulative count over 20 repeats. Since the same 115 final sequences are evaluated repeatedly, these counts describe prediction stability across training selections and must not be interpreted as 2,300 independent test samples. The corresponding KNN, RandomForest, and MLP matrices are provided as supplementary figures.

The supplementary material additionally reports the per-cell mean and sample standard deviation across the 20 repeat-level row-normalized confusion matrices. This measures sensitivity to few-shot training selection. It is distinct from the actuator-space temporal stability evaluated in Section 5.1.

### 5.4 Compact refinement versus the full actuator state

Relative to Refined ORCA-17, Compact Refined-7 improves final-test accuracy for SVM, KNN, and MLP, while RandomForest is approximately unchanged. The result indicates that not every actuator contributes equally to sequence discrimination under limited supervision. Removing inactive, unstable, or redundant channels can reduce estimation variance without requiring a learned feature selector at test time.

### 5.5 Dimension-controlled supporting results

Flexion-only and PCA-controlled representations provide supporting evidence that dimensionality alone does not explain performance. PCA can preserve discriminative variance, and the flexion subsets are strong; however, Compact Refined-7 combines lower dimension with explicit semantic coverage and a frozen development-only selection rule. The purpose of this comparison is not to claim universal superiority over PCA, but to distinguish embodiment-aware compactness from generic projection.

This dimension-controlled comparison answers RQ1: the actuator projection is competitive as a bounded semantic representation, but it is not claimed to dominate every statistical or geometric representation for every classifier.

### 5.6 Ablation and runtime

Removing temporal regularization increases mean actuator velocity and acceleration. Replacing Huber fitting with L2 notably reduces classification performance in the supporting ablation, consistent with sensitivity to large residuals. Palm-normal removal has a smaller effect in the current dataset and should not be overstated.

The optimizer processes 300 diagnostic frames in 27.61 ms on average, with a median of 27.27 ms and a 95th percentile of 32.62 ms. Mean iterations are 6.02, and both optimization success and finite-output rates are 1.0. These measurements support the term **causal frame-wise refinement**, not a hardware-independent real-time claim.

## 6. Discussion

### 6.1 What each stage contributes

The frame-wise projection and temporal optimizer solve different problems. The projection is a strong structured representation because it converts raw coordinates into bounded semantic articulation variables. The optimizer does not need to dominate it in every classifier to be meaningful. Its primary demonstrated role is to reduce temporal variation and corruption sensitivity. Compact selection then removes channels that are less useful under the target recognition protocol.

This decomposition resolves an apparent contradiction in earlier experiments where Actuator Projection-17 occasionally classified better than Refined ORCA-17. Classification and smoothness are different objectives. Temporal regularization can suppress informative amplitude as well as noise, and a full 17-dimensional state may retain redundant channels. Compact Refined-7 recovers a better trade-off without changing the frozen optimizer.

### 6.2 Why refinement is not ordinary smoothing

Coordinate filters operate independently or locally on landmark trajectories. The proposed optimizer instead searches a bounded actuator state whose predicted points are generated by an articulated MuJoCo model. It combines observation evidence with embodiment, initialization, and temporal priors. Nevertheless, conventional filters remain valid and sometimes stronger smoothness baselines. The contribution is a structured latent representation, not the claim that MuJoCo always produces the numerically smoothest coordinates.

### 6.3 Interpretation of the final classifier results

SVM and KNN benefit most from Compact Refined-7, suggesting that the compact actuator geometry creates useful margins and neighborhoods. RandomForest is less sensitive to the compact refined representation and slightly favors JointAngle-11 in mean accuracy, although the difference is not significant. MLP shows a positive but non-significant difference. Reporting these outcomes together is more credible than selecting a single favorable classifier.

### 6.4 Scientific scope

The present paper establishes a compact model-constrained representation pipeline. It does not yet implement explicit confidence-aware occlusion handling. Huber fitting reduces the influence of large residuals, but an occluded landmark still contributes to the observation objective. Landmark-wise reliability, missing-observation masks, and learned temporal priors are reserved for future work rather than added after final-test inspection.

## 7. Limitations

1. The current dataset does not establish subject-independent generalization; a multi-participant, multi-session evaluation is needed.
2. There is no synchronized ground-truth 3D human hand pose. Stability and controlled corruption sensitivity must not be interpreted as anatomical reconstruction accuracy.
3. ORCA is a robotic embodiment and does not exactly match human bone lengths or joint axes. Reconstructed ORCA Landmarks can therefore contain systematic reconstruction bias.
4. The acquisition and model convention is right-hand specific and does not constitute general left/right canonicalization.
5. Compact7 was selected on the frozen development set and should be validated on an external dataset before being treated as universal.
6. JointAngle-11 follows a published geometric definition but is not a full reproduction of the source paper's complete system.
7. The primary downstream encoder uses linear Resample16. It preserves coarse temporal order but does not replace a learned long-range temporal model.
8. The controlled corruption benchmark approximates observation failures and cannot reproduce every real occlusion pattern.

## 8. Conclusion

This study presents a compact ORCA actuator representation with MuJoCo-constrained causal temporal refinement for few-shot Chinese dance gesture recognition. The evidence supports three specific conclusions. First, the ORCA actuator projection provides an interpretable structured representation of MediaPipe landmarks. Second, causal MuJoCo refinement substantially reduces actuator-space temporal variation and sensitivity to controlled landmark corruption. Third, a seven-actuator subset frozen using development data reduces encoded dimensionality and improves frozen final-test performance over JointAngle-11 for SVM and KNN, while RandomForest and MLP differences are not significant after correction. The method should therefore be understood as a compact, model-consistent representation and refinement framework, not as ground-truth 3D hand recovery or a universally superior classifier input.

## Submission Statements To Complete

- **Author Contributions:** [TODO: use the Sensors CRediT-style statement and list who designed, coded, collected data, analyzed results, and wrote the manuscript].
- **Institutional Review Board Statement:** [TODO: approval body and identifier, or an accurate exemption/waiver statement].
- **Informed Consent Statement:** [TODO: state how participant consent and permission to publish identifiable images/video were obtained].
- **Data Availability Statement:** [TODO: repository link, controlled-access procedure, or a justified restriction].
- **Funding:** [TODO: funding body and grant, or “This research received no external funding.”].
- **Conflicts of Interest:** [TODO: declaration].
- **Correspondence:** [TODO: affiliation, institutional email, and corresponding author].

## Figure Plan

1. Final pipeline.
2. Full 17-actuator inventory and frozen Compact Refined-7 subset.
3. Representative actuator trajectory selected by a documented median-positive rule.
4. Aggregate actuator-space temporal stability.
5. Dimension-controlled representation comparison.
6. Development-only choice of `K=7`.
7. Frozen Compact Refined-7 versus JointAngle-11 with Holm-corrected significance.
8. Recognition performance versus representation dimension.
9. Controlled perturbation sensitivity.
10. Aggregate SVM confusion comparison on the frozen final test.

## Table Plan

1. ORCA actuator definitions.
2. Frozen Compact Refined-7 actuator subset.
3. Representation dimensions and Resample16 encoded sizes.
4. Frozen final classification results across four classifiers.
5. Paired Compact Refined-7 versus JointAngle-11 statistics.
6. Actuator-space temporal stability.
7. Controlled perturbation sensitivity.
8. Supporting loss ablation.
9. Runtime and convergence diagnostics.
10. Frozen optimizer parameters.
11. Frozen classifier parameters.
12. Controlled corruption protocol.

## References

The LaTeX version uses the project bibliography in `references.bib`, including MediaPipe Hands, MuJoCo, MANO, Huber loss, Savitzky-Golay, One Euro, Kalman filtering, PCA, few-shot learning, and gesture-recognition references.
