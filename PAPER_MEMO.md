# 论文备忘录

## 暂定题目

**Robust Hand Landmark Correction via MuJoCo-Constrained Temporal Optimization**

这个标题目前最稳，原因是：

- `Robust` 对应当前问题里的漂移、抖动和瞬时异常跳变
- `Hand Landmark Correction` 明确了输入输出对象是 noisy hand landmarks，而不是泛化的 gesture semantics
- `MuJoCo-Constrained Temporal Optimization` 准确描述了方法核心：不是普通滤波，而是在 ORCA actuator latent space 中做受 MuJoCo forward kinematics 约束的时序优化

相比 `gesture denoising`，`landmark correction` 更精确，也更贴近当前代码和实验，因为它强调的是受结构约束的修正，而不是普通滤波。

## 摘要开头建议

下面这句可以直接作为英文摘要中的关键 framing：

**Unlike conventional temporal smoothing, the proposed correction process is performed in an ORCA actuator space and constrained by MuJoCo forward kinematics, enabling physically plausible and temporally consistent refinement of noisy hand observations.**

如果想更完整一点，也可以用这一段作为摘要前两句草稿：

**Vision-based hand landmark estimators such as MediaPipe often suffer from frame-level jitter, drift, and transient outliers under challenging viewpoints and self-occlusion. To address this issue, we propose a MuJoCo-constrained temporal optimization framework that corrects noisy hand landmarks in an ORCA actuator space, producing physically plausible and temporally consistent latent hand trajectories.**

## 当前核心问题

本项目研究的问题不是“机械手能不能模仿人手动作”，而是：

**MediaPipe 手部关键点在特定角度下会产生抖动、漂移和瞬时异常跳变时，如何利用 ORCA 机器人手结构先验和 MuJoCo 前向模型获得更稳定、更适合 few-shot 手势识别的表示？**

也就是说，我们关心的是：

- 原始 MediaPipe landmarks 是否足够稳定
- ORCA actuator-space 是否能作为更好的低维结构化表示
- MuJoCo 约束优化能否进一步缓解瞬时漂移
- 更平滑的 latent actuator representation 是否能提升下游 few-shot sequence classification

## 推荐论文结构

当前最推荐采用紧凑版结构，把“单帧修复”作为主方法，把“sequence aggregation + classifier”明确放在下游评估位置：

### 1. Introduction

### 2. Related Work

### 3. Method

#### 3.1 Raw hand landmark extraction

#### 3.2 Frame-level physical correction

#### 3.3 Optimized action representation

#### 3.4 Sequence-level statistical aggregation for downstream evaluation

这里要特别强调：

- `frame-level physical correction` 是主方法
- `mean / std / max / delta` 是下游 sequence classification 的评估协议
- sequence aggregation 不是单帧修复本体

### 4. Experiments and Results

#### 4.1 Dataset and gesture classes

#### 4.2 Frame-level stability evaluation

#### 4.3 Sequence-level few-shot classification

#### 4.4 Classifier comparison

#### 4.5 Ablation study

### 5. Discussion

#### 5.1 Why frame correction improves sequence classification

#### 5.2 Reconstruction vs. discriminative representation

#### 5.3 Limitation of statistical aggregation

#### 5.4 Future work with stronger temporal models

### 6. Conclusion

这个结构的优点是：

- 主次清楚：先讲单帧修复，再讲下游分类验证
- 论文故事紧凑：不把统计聚合误写成主方法
- 审稿逻辑顺：先证明更稳，再证明更好分类
- 便于以后扩展：后续如果补 GRU/LSTM baseline，可以自然放进 Discussion 或新的实验小节

## 当前方法结构

目前已经有以下表示：

- `raw`
  - MediaPipe 归一化后的 `21x3` 关键点
- `geom`
  - 从关键点中提取的几何特征
- `corrected`
  - 基于 ORCA actuator 空间的低维结构校正特征
- `optimized_action`
  - 当前正式版本的 MuJoCo 优化得到的 ORCA latent actuator state，加入 Huber landmark loss 和 acceleration temporal regularization
- `optimized_full`
  - 将 `optimized_action` 再投影回 landmark 空间后得到的完整重建点坐标

## 为什么当前先采用统计型时序方法

当前使用的 `mean / std / max / delta` 不是主方法，而是：

**第一阶段最轻量、最可控的 downstream sequence evaluation protocol。**

这样做有三个原因：

1. 可以把论文主贡献集中在 `frame-level correction` 上，而不是被更复杂的时序分类器掩盖
2. 在当前 few-shot 数据规模下，固定长度统计表示具有更好的 bias--variance tradeoff，更不容易因为高容量时序模型而过拟合
3. 如果在这样一个简单下游协议下，`optimized_action` 仍然明显优于 `raw`、`corrected` 和 PCA baseline，那么就更能说明提升来自表示质量本身

因此，当前统计型时序处理的定位应该是：

- 不是“最强时序模型”
- 而是“用于隔离和验证 frame-level representation quality 的轻量评估协议”

这也是为什么第一篇论文可以先使用这一套，而后续再补：

- `GRU`
- `LSTM`
- temporal CNN
- transformer baseline

## 实验 baseline 体系应该如何划分

当前论文后续最需要补强的，不是再随意增加模型，而是把 baseline 体系分清楚。最稳妥的做法是把 baseline 分成下面几类：

### 1. Raw baseline

- `raw`

这是一切比较的原始起点，用来回答：

**不做任何结构约束和时序修正时，MediaPipe landmarks 本身能达到什么水平。**

### 2. Dimensionality reduction baseline

- `PCA-17`
- `best PCA`（例如当前 sweep 中最优的 `PCA-12`）

这一类 baseline 回答的是：

**当前方法的提升是不是只是因为从 63 维降到了更低维。**

也就是说，PCA 检验的是：

- 低维化本身的贡献
- 统计降维能否解释当前提升

因此，PCA 不能丢，而且必须继续保留。

### 3. Temporal smoothing / filtering baseline

- moving average
- Savitzky-Golay
- One-Euro
- Kalman

这一类 baseline 回答的是：

**当前方法的提升是不是只是因为对 noisy landmark trajectory 做了更复杂的平滑。**

也就是说，这一组检验的是：

- 时间平滑本身的贡献
- 普通 landmark-space denoising 是否已经足够

### 4. Structured projection baseline

- `corrected`

这一类回答的是：

**如果只利用 ORCA 结构先验做启发式 actuator-space projection，不做 MuJoCo 时序优化，会达到什么效果。**

### 5. Constrained refinement method

- `optimized_action`
- `optimized_full`

这才是当前论文的核心方法输出。

其中：

- `optimized_action` 是主表示
- `optimized_full` 是 forward reconstruction 后的 landmark-space 输出

## 为什么 PCA 和 smoothing 都必须做

PCA 和 smoothing 不是同一类 baseline，不能互相替代。

### PCA 解决的问题

PCA 主要回答：

**你这个方法是不是只是因为降维。**

### smoothing 解决的问题

smoothing / filtering 主要回答：

**你这个方法是不是只是另一种更复杂的平滑。**

因此，论文最终必须同时回答两个问题：

1. 当前方法不只是降维
2. 当前方法也不只是平滑

这也是为什么：

- `PCA-17 / best PCA` 必须保留
- `moving average / SG / One-Euro / Kalman` 也必须补上

## 最推荐的实验组织方式

### A. Landmark space

主要比较：

- `raw`
- `moving average`
- `Savitzky-Golay`
- `One-Euro`
- `Kalman`
- `optimized_full`

这里主要看：

- landmark-space temporal stability
- downstream classification

### B. Actuator / latent space

主要比较：

- `corrected`
- `PCA-17`
- `best PCA`
- `optimized_action`

这里主要看：

- actuator-space temporal stability
- downstream classification

## 一句话总结

最清楚的说法是：

**PCA 是 dimensionality baseline，smoothing/filtering 是 temporal denoising baseline，而 `corrected` / `optimized_action` / `optimized_full` 是结构约束方法本身。论文必须同时证明：当前方法不只是降维，也不只是平滑。**

## 为什么 smoothing baseline 仍然必须比较

这一部分很重要，需要专门记录清楚：

**和 smoothing baseline 比较，不是为了说明 `optimized_action` 和 Kalman / One-Euro 是同一种方法，而是为了证明当前方法的提升不能被“普通时间平滑”解释掉。**

也就是说，这个比较回答的问题不是：

- 谁是同类 filtering algorithm 里最强的

而是：

- 当前方法的提升，是否只不过来自某种简单 temporal smoothing

答案从当前实验看是：

**不是。**

### 为什么这个比较仍然有意义

虽然：

- `moving average`
- `Savitzky-Golay`
- `One-Euro`
- `Kalman`

都属于 landmark-space smoothing / filtering 方法，

而：

- `corrected`
- `optimized_action`
- `optimized_full`

属于 ORCA/MuJoCo 结构约束表示，

它们不是同一种算法范式，但它们都可以作为：

**同一个下游 gesture recognition 任务的候选输入表示。**

因此需要分清两种比较：

### 1. 同空间公平比较

如果问题是：

**“在 landmark space 里，只做时序平滑是否已经足够？”**

那么应该比较：

- `raw`
- `moving_average_raw`
- `savgol_raw`
- `oneeuro_raw`
- `kalman_raw`
- `optimized_full`

这里之所以用 `optimized_full`，不是因为它是最终最优表示，而是因为：

**`optimized_full` 是当前方法回到 landmark space 的输出，因此它可以和传统 landmark-space smoothing baseline 做更公平的比较。**

这一组比较回答的是：

**即使把当前方法重新投影回 landmark 坐标空间，它是否仍然优于普通平滑？**

如果答案是是的，那么就能说明：

**当前方法不是普通 smoothing 的复杂改写版。**

### 2. 最终任务表现比较

如果问题是：

**“哪一种表示最适合下游 few-shot dynamic gesture classification？”**

那么可以比较：

- `raw`
- `PCA-17`
- `best PCA`
- `corrected`
- `optimized_action`
- `optimized_full`

这时比较的不是“同类算法”，而是：

**不同表示在同一个下游任务上的最终有效性。**

在这一层面，最关键的发现是：

**`optimized_action` 是当前最强的最终表示。**

### 当前 smoothing baseline 的实验结论

在当前 `6-class Chinese dance` 数据集、`RandomForest` 分类器下，结果为：

- `raw`: `0.8708 ± 0.0617`
- `moving_average_raw`: `0.8667 ± 0.0890`
- `savgol_raw`: `0.8667 ± 0.0808`
- `oneeuro_raw`: `0.8792 ± 0.0853`
- `kalman_raw`: `0.9125 ± 0.0720`
- `corrected`: `0.9083 ± 0.0741`
- `optimized_action`: `0.9292 ± 0.0660`
- `optimized_full`: `0.8750 ± 0.0968`

这一组结果和 landmark-space jitter 结果合起来说明：

1. 普通 smoothing / filtering 在坐标空间中确实很有效，尤其 `Kalman` 是当前最强的 landmark-space smoothing baseline
2. 如果只看 velocity / acceleration 一类的平滑度指标，传统滤波器会比 `optimized_full` 更平
3. 但分类表现最好的并不是最平滑的 landmark-space baseline，而是 `optimized_action`
4. 因此本文方法的优势不能写成“最强 landmark smoother”，而应该写成“更适合识别的结构化表示”

也就是说，当前更准确的论文表述是：

**Although conventional smoothing reduces landmark-space jitter more aggressively than the reconstructed MuJoCo output, the best downstream recognition result is obtained by the optimized actuator representation rather than by the smoothest landmark-space baseline.**

或者更完整地写成：

**Ordinary coordinate-space smoothing and structured actuator-space refinement solve related but different problems: the former directly minimizes landmark variation, whereas the latter aims to preserve a temporally stable, embodiment-consistent, and discriminative latent hand representation.**

### 最关键的逻辑结论

这一组 baseline 的意义不是证明：

- `optimized_action` 和 Kalman 是同一种方法

而是证明：

**当前方法的效果不是“做了更复杂的平滑”这么简单。**

因此论文里最安全的表述应该是：

- smoothing baseline comparison 用来排除 “只是普通平滑” 这一解释
- PCA baseline comparison 用来排除 “只是普通降维” 这一解释
- `optimized_action` 的最优结果则说明：真正有效的是 ORCA/MuJoCo 结构约束下的 latent actuator refinement

### 这部分在论文里应该怎么组织

最稳妥的组织方式是：

#### Table A: Landmark-space smoothing comparison

放：

- `raw`
- `moving average`
- `Savitzky-Golay`
- `One-Euro`
- `Kalman`
- `optimized_full`

这张表的目的：

**证明传统 smoothing 在 landmark space 中确实非常有效，但“最平滑的坐标序列”并不等于“最适合 few-shot gesture recognition 的表示”。**

#### Table B: Representation comparison for downstream recognition

放：

- `raw`
- `PCA-17`
- `best PCA`
- `corrected`
- `optimized_action`
- `optimized_full`

这张表的目的：

**证明当前最优表示不是 raw，也不是 generic PCA，而是结构约束下得到的 `optimized_action` actuator latent representation。**

### 当前一句话记录

这部分最值得记住的一句话是：

**Smoothing baseline comparison does not claim algorithmic homogeneity; instead, it serves to show that the benefit of the proposed method cannot be reduced to ordinary temporal smoothing.**

## 论文中的论证顺序应该怎么写

这一部分也需要固定下来，因为它决定了整篇论文的 Results / Discussion 逻辑。

当前最合理、最稳的论证顺序是：

### 第一步：先排除 “只是 smoothing” 的解释

首先比较：

- `raw`
- `moving average`
- `Savitzky-Golay`
- `One-Euro`
- `Kalman`
- `optimized_full`

这里的重点不是证明 `optimized_full` 是全局最优表示，而是回答：

**当前方法的提升，是否仅仅来自某种普通时间平滑？**

如果 `optimized_full` 明显优于这些传统 smoothing/filtering baseline，那么可以得到第一层结论：

**the gain cannot be explained by ordinary temporal smoothing alone**

也就是说：

- 当前方法不是简单 low-pass filtering
- 不是把 landmark 轨迹做得更平一点就结束了
- 即使回到 landmark space，它仍然比传统 smoothing 更强

这是第一层排除性论证。

### 第二步：再排除 “只是降维” 的解释

在排除普通 smoothing 之后，下一步再比较：

- `raw`
- `PCA-17`
- `best PCA`
- `corrected`
- `optimized_action`
- `optimized_full`

这里要回答的问题是：

**当前方法的优势，是否只是因为把 noisy 高维 landmarks 压缩成了更低维的表示？**

如果 `corrected` / `optimized_action` 仍然优于 PCA baseline，那么可以得到第二层结论：

**the gain is not merely due to generic dimensionality reduction**

也就是说：

- 当前方法的优势不只是 feature 更短
- 不只是 small-sample setting 下低维更容易分类
- 真正有效的是结构约束下的表示方式

这是第二层排除性论证。

### 第三步：最后推出真正的核心解释

当：

- “只是 smoothing”
- “只是 PCA / 降维”

这两种替代解释都被排除之后，

就可以自然推出最终主张：

**性能提升更合理地归因于 ORCA embodiment prior、MuJoCo-constrained actuator-space refinement，以及 temporal regularization 的共同作用。**

也就是说，最后真正要强调的不是：

- 我们用了一个更复杂的滤波器
- 我们用了一个更低维的特征

而是：

**我们在结构受限的 actuator latent space 中，对 noisy MediaPipe observations 做了约束优化，从而得到了更适合下游 few-shot gesture recognition 的表示。**

### 为什么这个顺序最合理

这个顺序的优点是，它不是在机械地“堆 baseline”，而是在做：

**progressive elimination of alternative explanations**

也就是逐层排除：

1. 不是普通平滑
2. 不是普通降维
3. 因而更可能是结构约束 refinement 真正起作用

这个逻辑对审稿人非常友好，因为它清楚回答了：

- 为什么要做 smoothing baseline
- 为什么还要做 PCA baseline
- 为什么最后还能回到 `optimized_action` 的主结论

### 论文里可以直接用的过渡句

从 smoothing baseline 过渡到 PCA baseline，可以写成：

**After showing that the proposed method is not reducible to conventional landmark-space smoothing, we further examine whether its advantage could simply arise from dimensionality reduction.**

从 PCA baseline 过渡到最终结论，可以写成：

**The results show that neither conventional smoothing nor generic dimensionality reduction fully explains the observed gains. Instead, the strongest performance is obtained from the MuJoCo-refined actuator representation, suggesting that structured latent-state refinement is the main source of improvement.**

### 当前最值得记住的一句话

**先证明不是 simple smoothing，再证明不是 generic dimensionality reduction，最后再论证 ORCA/MuJoCo 结构约束优化才是性能提升的关键来源。**

## 研究思路演化应该怎么表述

这一部分也建议固定下来，因为它能把“做 baseline”和“提出方法”之间的关系解释得更自然。

当前最准确的说法不是：

- 先随便做了一个 smoothing
- 然后又做了一个 action 方法

而是：

**我们在系统地回答一个研究问题：MediaPipe 的抖动与漂移，是否可以仅靠普通 temporal smoothing 解决；如果不能，那么结构约束的 actuator-space refinement 是否更有效。**

因此，研究思路应该写成下面这个顺序：

### 1. 从 raw MediaPipe 出发

原始 `raw landmarks` 存在：

- jitter
- drift
- transient outliers
- 局部结构不一致

这意味着下游动态手势识别可能受到 noisy observations 的影响。

### 2. 先用 smoothing baseline 检验一个更简单的解释

第一步并不是直接提出复杂方法，而是先问：

**如果这些误差只是普通时间噪声，那么简单 smoothing/filtering 是否已经足够？**

因此引入：

- moving average
- Savitzky-Golay
- One-Euro
- Kalman

这些 baseline 的目的不是作为论文主方法，而是作为：

**simpler alternative explanations**

如果它们已经足够好，那么结构约束优化的必要性就会减弱。

### 3. 结果表明 smoothing 只有有限帮助

当前结果说明：

- smoothing 相比 `raw` 有一定提升
- 但提升有限
- 并不能达到结构约束方法的效果

因此可以推断：

**MediaPipe 的误差并不只是高频时间噪声，而更可能包含结构不一致、局部错误和瞬时异常观测。**

这一步非常关键，因为它给出了引入 actuator-space refinement 的动机。

### 4. 再引入结构约束表示与优化

在确认“普通 smoothing 不够”之后，再进一步引入：

- `corrected`
- `optimized_action`
- `optimized_full`

这一步的逻辑是：

**既然问题不只是 temporal fluctuation，那么就需要利用 ORCA embodiment prior 和 MuJoCo forward-kinematic constraints，在结构受限的 actuator space 中对观测进行修正。**

### 5. 最终发现 optimized_action 最强

实验结果表明：

- `optimized_action` 是当前最好的下游表示
- `optimized_full` 作为 landmark-space 输出也优于普通 smoothing

因此最终可以得出更强的解释：

**真正有效的不是“平滑得更厉害”，而是通过结构约束 latent-state refinement 对 noisy observations 进行了更合理的修正。**

### 当前最简洁的一句话

这部分最值得记录的一句话是：

**We first tested whether conventional temporal smoothing was sufficient to suppress MediaPipe noise. The limited gains from smoothing baselines motivated the use of ORCA/MuJoCo-constrained actuator-space refinement, which ultimately produced the strongest downstream representation.**

## 建议的 baseline 实验矩阵

下面这张矩阵可以直接作为后续补实验的执行顺序参考。

| 类别 | 方法 | 所在空间 | 主要用途 | 应优先比较的指标 |
|---|---|---|---|---|
| Raw baseline | `raw` | landmark space | 原始观测起点 | jitter + classification |
| Smoothing baseline | moving average | landmark space | 最简单平滑 | jitter + classification |
| Smoothing baseline | Savitzky-Golay | landmark space | 局部多项式平滑 | jitter + classification |
| Smoothing baseline | One-Euro | landmark space | 实时 tracking 平滑 | jitter + classification |
| Smoothing baseline | Kalman | landmark space | 时序状态估计 | jitter + classification |
| Structured method output | `optimized_full` | landmark space | 结构约束后的重建 landmarks | jitter + classification |
| Dimensionality baseline | `PCA-17` | latent / feature space | 维度匹配 baseline | classification |
| Dimensionality baseline | `best PCA` | latent / feature space | 最优统计降维 baseline | classification |
| Structured projection | `corrected` | actuator space | 启发式 ORCA 映射 | jitter + classification |
| Constrained refinement | `optimized_action` | actuator space | MuJoCo 优化 latent state | jitter + classification |

## 最推荐的比较方式

### A. Landmark-space 比较

这一组回答：

**如果只在 landmark sequence 上做时序平滑，能否达到类似效果？**

建议比较：

- `raw`
- `moving average`
- `Savitzky-Golay`
- `One-Euro`
- `Kalman`
- `optimized_full`

建议指标：

- velocity mean / rms
- acceleration mean / rms
- downstream few-shot classification

### B. Actuator-space 比较

这一组回答：

**如果进入 ORCA actuator latent space，MuJoCo refinement 是否比 heuristic projection 更稳、更有判别性？**

建议比较：

- `corrected`
- `optimized_action`

如果后面做 ablation，可以继续加：

- `optimized_action_no_huber`
- `optimized_action_no_acceleration`
- `optimized_action_no_temporal`

建议指标：

- actuator-space velocity / acceleration
- downstream few-shot classification

### C. 降维 baseline 比较

这一组回答：

**当前方法是不是只是因为低维化。**

建议比较：

- `raw`
- `PCA-17`
- `best PCA`
- `corrected`
- `optimized_action`

建议指标：

- downstream few-shot classification
- 必要时补充 latent-space stability

## 建议的执行顺序

如果按工作量和收益排序，最推荐的执行顺序是：

1. `moving average`
2. `Savitzky-Golay`
3. `One-Euro`
4. `Kalman`
5. 补 landmark-space jitter + classification
6. 组织 actuator-space `corrected` vs `optimized_action` 的 jitter 和 classification
7. 保留 `PCA-17` + `best PCA` 作为降维对照
8. 最后再考虑 `GRU / LSTM` 这类 learned temporal denoising baseline

## 为什么这个矩阵重要

这张矩阵最终可以帮助论文回答四个不同的问题：

1. `raw` 本身有多不稳定？
2. 普通时序平滑能改善多少？
3. 纯统计降维能改善多少？
4. ORCA + MuJoCo 结构约束优化能否在这些 baseline 之上继续提升？

来进一步验证 refined representation 对更强 temporal models 是否同样有帮助。

## 各类表示与数学内涵

### 1. `corrected`

`corrected` 是一个基于规则的 ORCA actuator-space projection。它不是优化结果，也不是学习出来的 embedding。

输入是 MediaPipe landmarks：

\[
\mathbf{y}_t \in \mathbb{R}^{21 \times 3}
\]

也可以展平成 63 维：

\[
\mathbf{y}_t \in \mathbb{R}^{63}
\]

`corrected` 从这些点中提取手工几何语义，例如：

- 手腕方向
- 手指弯曲程度
- 手指张开程度
- 拇指打开程度
- 掌面法向

然后映射到 ORCA 的 17 维 actuator 空间：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t), \qquad
\tilde{\mathbf{q}}_t \in \mathbb{R}^{17}
\]

其中 \(g(\cdot)\) 是手工设计的几何映射函数。

它的核心数学意义是：

**用结构化 actuator 变量替代高维 landmark 坐标。**

原始 landmark 描述的是：

\[
(x_1,y_1,z_1,\dots,x_{21},y_{21},z_{21})
\]

而 `corrected` 描述的是：

\[
[
q_{wrist},
q_{pinky-abd},
q_{pinky-mcp},
q_{pinky-pip},
\dots,
q_{thumb-pip}
]
\]

所以 `corrected` 是一种：

**semantic geometric projection**

或者：

**embodiment-constrained reparameterization**

它的优势是低维、可解释、结构受限，适合 few-shot 分类。它的局限是逐帧计算：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

也就是说，它没有显式利用上一帧或下一帧，因此不是严格意义上的 temporal anti-jitter filter。

### 2. `optimized_action`

`optimized_action` 是当前正式版本的 MuJoCo 优化表示，也是目前最有价值的表示。

它仍然输出 17 维 actuator latent state：

\[
\mathbf{q}_t^* \in \mathbb{R}^{17}
\]

它的两个关键机制是：

1. Huber landmark loss
2. acceleration temporal regularization

因此它更接近一个：

**robust temporally regularized latent-state estimator**

当前目标函数为：

\[
\mathbf{q}_t^*
=
\arg\min_{\mathbf{q}_t \in \mathcal{Q}}
\lambda_l \mathcal{L}_{huber}
+ \lambda_n \mathcal{L}_{normal}
+ \lambda_p \mathcal{L}_{prior}
+ \lambda_s \mathcal{L}_{temporal}
+ \lambda_a \mathcal{L}_{acceleration}
+ \lambda_d \mathcal{L}_{default}
+ \lambda_b \mathcal{L}_{boundary}
\]

Huber loss 的作用是降低异常 MediaPipe 点的影响。对残差 \(r\)，Huber loss 可以写为：

\[
\rho_\delta(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r| \le \delta \\
\delta(|r| - \frac{1}{2}\delta), & |r| > \delta
\end{cases}
\]

它的特点是：

- 小误差时像 L2，正常拟合
- 大误差时像 L1，降低 outlier 的影响

因此，如果某个 MediaPipe 点突然漂得很远，Huber loss 不会像普通 L2 loss 那样被异常点强烈拉走。

新增的 acceleration loss 是：

\[
\mathcal{L}_{acceleration}
=
\|\mathbf{q}_t - 2\mathbf{q}_{t-1}^* + \mathbf{q}_{t-2}^*\|_2^2
\]

这是离散二阶差分，对应状态轨迹的加速度或突然转折。它正好用于惩罚 MediaPipe 一闪而过造成的突变。

因此 `optimized_action` 的数学内涵是：

**在 ORCA actuator latent space 中进行鲁棒时序状态估计。**

它同时利用：

- 当前观测
- ORCA 结构先验
- MuJoCo 前向运动学
- 上一帧状态
- 上上帧状态
- 默认姿态
- actuator 边界

相比 `corrected` 的逐帧映射：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

`optimized_action` 是：

\[
\mathbf{q}_t^*
=
\arg\min
\mathcal{L}(\mathbf{q}_t, \mathbf{y}_t, \mathbf{q}_{t-1}^*, \mathbf{q}_{t-2}^*)
\]

因此它更接近真正的 tracking / filtering 方法。

### 3. `optimized_full`

`optimized_full` 不是新的优化变量，而是把 `optimized_action` 得到的 latent state 重新投影回 landmark 空间。

先得到：

\[
\mathbf{q}_t^*
\]

然后通过 MuJoCo forward kinematics 得到：

\[
\mathbf{y}_t^*
=
h(\mathbf{q}_t^*)
\]

其中：

- `optimized_action` 是 17 维 latent actuator state
- `optimized_full` 是 63 维 reconstructed landmark representation

所以 `optimized_full` 是一种：

**structure-consistent reconstructed landmark representation**

它适合做：

- 可视化
- 几何一致性评估
- temporal smoothness comparison
- refined landmark output

但它不一定最适合分类。原因包括：

1. 维度从 17 回到 63，few-shot 下更容易过拟合
2. 坐标空间会重新引入一些分类无关的几何变化
3. 它更偏 reconstruction，不一定更偏 discrimination

### 5. 四种表示的关系总结

| 表示 | 维度 | 来源 | 是否优化 | 是否时序 | 核心作用 |
|---|---:|---|---|---|---|
| `corrected` | 17 | 手工几何映射到 ORCA actuator | 否 | 否 | 结构化低维分类特征 |
| `optimized_action` | 17 | 鲁棒 MuJoCo 时序优化 | 是 | 较强时序 | 稳定且判别性强的 latent state |
| `optimized_full` | 63 | 经 MuJoCo forward 重建 | 间接 | 继承优化轨迹 | 结构一致重建 landmarks |

最核心的数学区别可以简写为：

\[
\textbf{corrected:}
\quad
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

\[
\textbf{optimized\_action:}
\quad
\mathbf{q}_t^*
=
\arg\min_{\mathbf{q}}
\left[
\rho_\delta(h(\mathbf{q})-\mathbf{y}_t)
+
\lambda_s\|\mathbf{q}-\mathbf{q}_{t-1}^*\|^2
+
\lambda_a\|\mathbf{q}-2\mathbf{q}_{t-1}^*+\mathbf{q}_{t-2}^*\|^2
+
\lambda_p\|\mathbf{q}-\tilde{\mathbf{q}}_t\|^2
\right]
\]

\[
\textbf{optimized\_full:}
\quad
\mathbf{y}_t^*
=
h(\mathbf{q}_t^*)
\]

因此，论文里最推荐的解释是：

**`corrected` 是 heuristic embodiment-aware projection；`optimized_action` 是 robust temporally regularized MuJoCo latent-state estimation；`optimized_full` 是 optimized latent state 的 forward-kinematic landmark reconstruction。**

## 方法理解

项目当前最重要的理论框架是：

1. MediaPipe landmarks 是带噪观测，不是真值
2. ORCA hand actuator space 提供低维、可解释、结构受限的 hand state
3. MuJoCo 提供前向运动学约束，用于评估候选 actuator state 是否能解释观测点
4. 对瞬时漂移问题，单帧结构映射不够，需要加入时序鲁棒优化
5. 对分类任务来说，最有效的表示不一定是完整重建点，而可能是低维 latent actuator state

## corrected 到底是什么

`corrected` 不是学习出来的 latent code，也不是 MuJoCo 优化结果。

它本质上是：

**从 MediaPipe landmarks 手工提取几何语义，再映射到 ORCA actuator 空间的低维结构化表示。**

流程为：

```text
MediaPipe 21x3 landmarks
-> normalize landmarks
-> extract hand geometric features
-> map features into ORCA actuator ranges
-> corrected 17D actuator-space representation
```

它的低维性来自 ORCA embodiment，而不是 PCA 这类通用降维算法。也就是说，`corrected` 是一种：

**embodiment-constrained reparameterization**

而不是普通 statistical dimensionality reduction。

## 优化方法的当前数学表达

设第 \(t\) 帧的观测为：

\[
\mathbf{y}_t \in \mathbb{R}^{21 \times 3}
\]

设 ORCA latent hand state 为：

\[
\mathbf{q}_t \in \mathbb{R}^{17}
\]

MuJoCo 前向映射为：

\[
\hat{\mathbf{y}}_t = h(\mathbf{q}_t)
\]

当前增强版优化目标为：

\[
\mathbf{q}_t^* = \arg\min_{\mathbf{q} \in \mathcal{Q}}
\lambda_l \mathcal{L}_{huber-landmark}
+ \lambda_n \mathcal{L}_{normal}
+ \lambda_p \mathcal{L}_{prior}
+ \lambda_s \mathcal{L}_{temporal}
+ \lambda_a \mathcal{L}_{acceleration}
+ \lambda_d \mathcal{L}_{default}
+ \lambda_b \mathcal{L}_{boundary}
\]

其中：

- `Huber landmark loss`
  - 降低异常 MediaPipe 点对优化的拉扯
- `normal loss`
  - 拟合掌面法向
- `prior loss`
  - 约束解不要偏离启发式 ORCA 投影太远
- `temporal loss`
  - 约束当前状态接近上一帧状态
- `acceleration loss`
  - 约束二阶时间变化，抑制一闪而过的跳变
- `default loss`
  - 防止过度偏离默认姿态
- `boundary loss`
  - 防止贴住 actuator 边界

新增的关键时序项是：

\[
\mathcal{L}_{acceleration}
=
\left\|
\mathbf{q}_t - 2\mathbf{q}_{t-1} + \mathbf{q}_{t-2}
\right\|_2^2
\]

最终重建后的优化点为：

\[
\mathbf{y}_t^* = h(\mathbf{q}_t^*)
\]

但当前结果显示，最适合分类的是 \(\mathbf{q}_t^*\) 这个 `optimized_action` latent state，而不是重建后的 \(\mathbf{y}_t^*\)。

## 当前实验设置

目前实验是：

- sequence-level classification
- few-shot setting
- `shots_per_class = 3`
- `repeats = 20`
- 分类器：`SVM`, `KNN`, `RandomForest`, `MLP`
- 类别：`6`, `7`, `8`
- 数据集：`gesture_sequence_dataset_optimized_v2.csv`

当前数据库已包含：

- 原始 `raw_*`
- `geom_*`
- `corrected_*`
- `optimized_action_*`
- `optimized_sparse_*`
- `optimized_full_*`
- `optimized_loss_*`
- `optimized_loss_acceleration`

## 当前关键实验结果

### 1. Temporal Jitter Evaluation

指标越低，表示时间上越平滑。

当前 v2 数据集上的结果：

- `raw`
  - `velocity_mean = 0.443881`
  - `acceleration_mean = 0.595201`
- `corrected`
  - `velocity_mean = 0.621456`
  - `acceleration_mean = 0.950898`
- `optimized_action`
  - `velocity_mean = 0.334345`
  - `acceleration_mean = 0.357958`
- `optimized_full`
  - `velocity_mean = 0.451672`
  - `acceleration_mean = 0.519420`

当前最重要的稳定性发现是：

**`optimized_action` 同时具有最低的 velocity 和 acceleration 指标，说明加入 Huber loss 与 acceleration temporal regularization 后，MuJoCo-constrained latent actuator state 明显降低了时间抖动。**

### 2. Few-Shot Sequence Classification

当前 v2 数据集上的结果：

- `optimized_action`: `0.8500 ± 0.1159`
- `corrected`: `0.8063 ± 0.1081`
- `optimized_full`: `0.7125 ± 0.1858`

当前排序为：

\[
optimized\_action > corrected > optimized\_full
\]

结合之前 raw baseline：

- `raw`: `0.6312 ± 0.1504`

因此当前总体趋势为：

\[
optimized\_action > corrected > optimized\_full > raw
\]

### 3. Multi-Classifier Comparison

后续补充的多分类器实验表明，当前观察并不是单一分类器偶然造成的。基于 `gesture_sequence_dataset_optimized_v2.csv`，`shots_per_class = 3`，`repeats = 20`：

#### SVM

- `raw`: `accuracy = 0.6313`, `macro_f1 = 0.6145`, `kappa = 0.4401`
- `corrected`: `accuracy = 0.8063`, `macro_f1 = 0.7808`, `kappa = 0.7025`
- `optimized_action`: `accuracy = 0.8500`, `macro_f1 = 0.8290`, `kappa = 0.7674`
- `optimized_full`: `accuracy = 0.7125`, `macro_f1 = 0.6723`, `kappa = 0.5675`

#### KNN

- `raw`: `accuracy = 0.5875`, `macro_f1 = 0.5392`, `kappa = 0.3595`
- `corrected`: `accuracy = 0.7875`, `macro_f1 = 0.7671`, `kappa = 0.6822`
- `optimized_action`: `accuracy = 0.8063`, `macro_f1 = 0.7802`, `kappa = 0.7114`
- `optimized_full`: `accuracy = 0.6375`, `macro_f1 = 0.6120`, `kappa = 0.4528`

#### RandomForest

- `raw`: `accuracy = 0.5938`, `macro_f1 = 0.5418`, `kappa = 0.3688`
- `corrected`: `accuracy = 0.8688`, `macro_f1 = 0.8679`, `kappa = 0.8030`
- `optimized_action`: `accuracy = 0.8938`, `macro_f1 = 0.8834`, `kappa = 0.8395`
- `optimized_full`: `accuracy = 0.7625`, `macro_f1 = 0.7480`, `kappa = 0.6371`

#### MLP

- `raw`: `accuracy = 0.6250`, `macro_f1 = 0.5934`, `kappa = 0.4188`
- `corrected`: `accuracy = 0.7625`, `macro_f1 = 0.7517`, `kappa = 0.6410`
- `optimized_action`: `accuracy = 0.8125`, `macro_f1 = 0.8013`, `kappa = 0.7148`
- `optimized_full`: `accuracy = 0.6875`, `macro_f1 = 0.6653`, `kappa = 0.5215`

当前最重要的新发现是：

- `optimized_action` 在四个分类器中都保持强竞争力
- `RandomForest + optimized_action` 是目前最强组合
- `corrected` 仍然稳定优于 `raw`
- `optimized_full` 通常不如 `optimized_action`，支持“低维结构化 latent representation 优于高维重建坐标”的论点

## 当前最重要的发现

当前最重要、最有论文价值的发现是：

**在 ORCA actuator 空间中加入 Huber robust landmark fitting 和 acceleration-based temporal regularization 后，MuJoCo-optimized latent actuator representation 不仅显著降低了 temporal jitter，而且在 few-shot sequence gesture classification 中超过了 heuristic corrected baseline。**

这个发现比早期结果更强，因为它同时支持两个主张：

- MuJoCo/ORCA 约束优化确实能缓解 MediaPipe 一闪而过的抖动与漂移
- 更稳定的 actuator-space latent state 能提升下游 few-shot 手势识别性能

## 当前现象的原理解释

目前最合理的解释是：

1. `raw`
   - 信息多，但噪声也大，few-shot 下容易被抖动和漂移拖累
2. `corrected`
   - 通过 ORCA 结构约束将高维 noisy landmarks 变成低维、语义明确、关节一致的表示，因此分类效果明显优于 raw
3. `optimized_action`
   - Huber loss 降低异常 landmark 的影响
   - acceleration loss 抑制二阶时间跳变
   - 因此同时获得更低 jitter 和更高分类准确率
4. `optimized_full`
   - 将 latent actuator state 再投影回高维点坐标，维度升高，few-shot 下仍然不如低维 actuator representation

## 当前适合写进论文的安全结论

目前最安全、最强的结论是：

**A robust temporally regularized MuJoCo optimization in ORCA actuator space produces a latent hand representation that reduces temporal jitter and improves few-shot sequence-level gesture classification compared with raw MediaPipe landmarks and heuristic corrected features.**

中文版可以写成：

**在 ORCA actuator 空间中进行带 Huber 观测项和二阶时间正则的 MuJoCo 约束优化，可以得到更平滑、更具判别性的 latent hand representation，从而同时缓解 MediaPipe 抖动并提升 few-shot sequence 手势识别性能。**

## 当前不适合夸大的结论

目前不要写成：

- full physical reconstruction
- exact hand pose recovery
- optimized full landmarks are the best representation
- MuJoCo dynamics fully solves MediaPipe drift

更准确的说法是：

- 当前方法是 MuJoCo forward-kinematics-based constrained optimization
- 当前最优表示是 optimized actuator latent state，而不是完整重建 landmarks
- 当前结果仍然需要更多数据和 ablation 支撑

## 当前论文最好的主线

当前最稳的论文主线是：

1. MediaPipe landmarks 有噪声、抖动和瞬时漂移
2. ORCA actuator space 提供低维结构化 hand state
3. 初始 corrected 表示证明了结构先验对 few-shot classification 有帮助
4. 进一步加入 MuJoCo forward fitting、Huber observation loss 和 acceleration temporal regularization
5. 得到的 `optimized_action` 同时降低 jitter，并提升 sequence-level few-shot classification
6. 结果说明：最有用的不是高维重建 landmarks，而是结构约束下的低维 latent actuator representation

## 为什么这个结果有意思

这个结果有意思，不只是因为准确率提升，而是因为它揭示了一个更深的 representation design 问题：

**对手势识别任务来说，最优表示未必是最原始的，也未必是最完整几何重建的，而可能是一个受 embodiment 约束、经鲁棒时序优化后的低维 latent actuator state。**

这实际上是一个关于：

- representation design
- embodiment prior
- robust temporal optimization
- few-shot robustness
- reconstruction vs discrimination tradeoff

的研究问题。

## 当前项目所处阶段

当前状态适合定义为：

- 已经有清晰的研究故事
- 已经有初步可重复的实验结果
- 已经有 jitter 指标和 classification 指标的双重证据
- 已经有多分类器对比结果
- 已经有 macro-F1 / precision / recall / kappa 指标
- 已经可以开始写论文 draft
- 但还需要补充更多实验才能形成更强投稿版本

## 当前不足

目前还存在这些不足：

- sequence 数量还不够大
- 类别数还比较少
- 还没有 PCA-17 baseline
- 还没有正式 ablation：Huber only / acceleration only / both
- 优化仍然是 sparse correspondence fitting，不是完整动力学反演
- 还没有真正的时序模型 baseline（例如 GRU / LSTM）
- 还没有跨 session 或跨采集条件泛化验证

## 方法定位与后续方向

这一部分非常重要，用来准确界定当前方法和最初研究愿景之间的关系。

### 1. 当前算法是不是对的

**对，是对的。**

而且它不是偏离最初想法很远，而是：

**原始想法的一个弱版本 / 初版实现。**

它已经具备三个核心元素：

- ORCA 低维结构先验
- MuJoCo forward kinematics 约束
- temporal regularization 抑制瞬时抖动

所以它不是错方向。

但它还没有完全做到最初最想要的那种形式：

**当手指被遮挡或观测失效时，主要依赖物理先验和时序先验去补全与修正。**

### 2. 当前方法更准确的定位

当前方法更准确的定位是：

**robust constrained correction under noisy observation**

而不是：

**explicit occlusion-aware physical hallucination / completion**

通俗一点说：

- 现在做的是“带先验的鲁棒修正”
- 后续真正想做的是“观测失效时的模型主导补全”

### 3. 为什么会觉得“还差一点”

这个直觉是对的，因为当前方法仍然把 MediaPipe 观测作为损失函数的重要组成部分。

如果某些手指被遮挡，而 MediaPipe 给出的点已经错得很离谱，那么：

- 即使使用了 Huber loss
- 它也只是降低异常观测的影响
- 但这些异常观测仍然没有被彻底从优化里拿掉

也就是说，当前方法已经能够：

- 减弱异常观测的拉扯
- 利用时序正则抑制一闪而过的跳变

但它还没有显式做到：

- 判断哪些点当前不可信
- 对不可信点降低权重
- 在缺失观测时主要依赖模型进行补全

### 4. 与最初研究愿景的最准确关系

最准确的表述是：

**最初的研究愿景是：利用 ORCA/MuJoCo 在视觉观测不可靠时修正甚至补全手部状态。当前实现已经实现了其中的第一步，即在 ORCA actuator space 中通过鲁棒观测项和时序正则进行约束优化，从而减弱异常观测的影响；但它尚未显式建模遮挡置信度或缺失观测。**

这句话非常适合写进 Discussion 或 Limitation，因为它同时说明了：

- 当前方法方向是对的
- 当前方法还不是最终完整版

### 5. 后续最值得推进的三个方向

#### A. 遮挡/置信度加权观测项

可以为不同关键点构造不同权重 \(w_{t,i}\)，例如依据：

- MediaPipe visibility / confidence
- 关键点几何一致性
- 帧间突变程度

把观测项写成：

\[
\mathcal{L}_{obs}
=
\sum_i w_{t,i}\,\rho_\delta(\|h_i(\mathbf{q}_t)-\mathbf{y}_{t,i}\|)
\]

这样被挡住、低置信度或明显突变的点会被自动降低权重。

#### B. 缺失观测 / 遮挡建模

如果某些点当前不可信，可以进一步考虑：

- 不拟合这些点
- 只使用可见点
- 再结合 ORCA/MuJoCo 结构先验与 temporal prior 进行补全

这会更接近“观测失效时由模型主导修正”的目标。

#### C. 更强的时序状态估计

当前仍然主要是逐帧优化加短时序正则。后续可以进一步考虑：

- joint sequence optimization
- Kalman-like latent filtering
- smoothing over a whole temporal window

这样会更强地利用“真实手部运动通常是连续且不会突然乱跳”的先验。

### 6. 一句话总结

最适合记录下来的判断是：

**当前的 `optimized_action` 和最初想法不是相反的，而是“目标一致、实现强度不同”。目标一致之处在于：都想用 ORCA/MuJoCo 修正 MediaPipe 抖动和遮挡误差；区别在于：最初想做的是更“观测失效感知”的版本，而当前实现的是一个更通用的鲁棒约束优化版本。**

因此：

**这不是方法错了，而是一个合理的第一篇论文版本；后续完全可以沿着“遮挡感知物理修正”方向继续推进。**

## 下一步必须做的实验

### 1. PCA-17 Baseline

目的：

确认 `corrected` / `optimized_action` 的优势不是单纯因为从 63 维降到了 17 维。

需要比较：

- `raw`
- `PCA(raw)-17`
- `corrected`
- `optimized_action`

如果：

\[
optimized\_action > corrected > PCA17 > raw
\]

那么论文论点会更强。

### 2. Macro-F1 and Weighted-F1

这一步已经完成。当前脚本已经输出：

- `accuracy_mean/std`
- `macro_f1_mean/std`
- `weighted_f1_mean/std`
- `macro_precision_mean/std`
- `macro_recall_mean/std`
- `cohen_kappa_mean/std`

这对小样本、多类别数据更公平。下一步更值得做的是把这些指标系统整理成论文总表和主结果图。

### 3. Ablation Study

需要比较：

- `corrected`
- `optimized_action`
- `optimized_action_no_huber`
- `optimized_action_no_acceleration`
- `optimized_full`

目的：

确认提升来自：

- Huber robust observation loss
- acceleration temporal regularization
- 二者组合

### 4. More Classifiers

这一步已经完成。当前已经比较：

- `SVM`
- `KNN`
- `RandomForest`
- `MLP`

结果说明当前趋势不是单一分类器偶然造成的。下一步可以考虑：

- 增加一个轻量时序 baseline
- 对分类器超参数做更系统的 sensitivity analysis

### 5. More Data

继续增加：

- 更多 sequence
- 更多类别
- 不同录制 session
- 不同角度和速度变化

### 6. Jitter / Classification Joint Table

最终论文最好有一张表同时展示：

- `velocity_mean`
- `acceleration_mean`
- `accuracy_mean`
- `macro_f1_mean`
- `kappa_mean`

这样可以证明方法同时提升：

- temporal stability
- downstream recognition

## 当前一句话版本

当前项目最值得记录的结论是：

**在 ORCA actuator 空间中进行鲁棒时序 MuJoCo 优化，可以得到比原始 MediaPipe landmarks 和启发式 corrected 表示更平滑、更适合 few-shot sequence 手势识别的 latent actuator representation。**

## `corrected` 与 `optimized_action` 的核心解释

这一部分是后续写论文 `Method` 时最重要的概念区分。

### 1. `corrected` 是什么

`corrected` 不是学习出来的 embedding，也不是 MuJoCo 优化结果。

它本质上是：

**把 noisy MediaPipe landmarks 按照 ORCA 机器手的结构语义，映射成一个 17 维 actuator-space 表示。**

设第 \(t\) 帧 MediaPipe 观测为：

\[
\mathbf{y}_t \in \mathbb{R}^{21 \times 3}
\]

或展开成：

\[
\mathbf{y}_t \in \mathbb{R}^{63}
\]

`corrected` 通过手工设计的几何映射函数 \(g(\cdot)\)，提取：

- 手腕方向
- 手指弯曲程度
- 手指张开程度
- 拇指打开程度
- 掌面法向

然后将它们映射到 ORCA actuator 空间：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t), \qquad \tilde{\mathbf{q}}_t \in \mathbb{R}^{17}
\]

因此，`corrected` 的本质不是普通 PCA 式降维，而是：

**embodiment-constrained reparameterization**

或者：

**semantic geometric projection into ORCA actuator space**

它的优点是：

- 从 63 维降到 17 维
- 表示更结构化
- 每一维有明确的 actuator / joint 语义
- few-shot 下更容易分类

它的局限是：

- 仍然是逐帧映射
- 本质仍是 \(\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)\)
- 没有真正求解最优物理状态
- 对单帧异常漂移的抑制能力有限

一句话总结：

**`corrected` 是基于 ORCA 结构先验的启发式低维映射。**

### 2. `optimized_action` 是什么

`optimized_action` 建立在 `corrected` 的基础上，但它不再是直接映射，而是：

**在 ORCA actuator latent space 中，通过 MuJoCo forward kinematics 做鲁棒约束优化，求得更可信的潜在手状态。**

设 ORCA latent hand state 为：

\[
\mathbf{q}_t \in \mathbb{R}^{17}
\]

MuJoCo 前向映射为：

\[
\hat{\mathbf{y}}_t = h(\mathbf{q}_t)
\]

其中 \(h(\cdot)\) 表示：给定一个 ORCA actuator state，MuJoCo 生成该状态下对应的 hand landmarks / sparse points。

于是 `optimized_action` 不是直接算：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

而是求解：

\[
\mathbf{q}_t^*
=
\arg\min_{\mathbf{q}_t \in \mathcal{Q}}
\mathcal{L}(\mathbf{q}_t)
\]

其中 \(\mathcal{Q}\) 是 actuator 的可行域。

### 3. `optimized_action` 的目标函数含义

当前增强版目标函数为：

\[
\mathbf{q}_t^* = \arg\min_{\mathbf{q}_t \in \mathcal{Q}}
\lambda_l \mathcal{L}_{huber}
+ \lambda_n \mathcal{L}_{normal}
+ \lambda_p \mathcal{L}_{prior}
+ \lambda_s \mathcal{L}_{temporal}
+ \lambda_a \mathcal{L}_{acceleration}
+ \lambda_d \mathcal{L}_{default}
+ \lambda_b \mathcal{L}_{boundary}
\]

各项含义如下：

#### 3.1 Huber landmark loss

这项让 MuJoCo 生成的 landmarks 去拟合 MediaPipe 观测，但对异常漂移点更鲁棒。

\[
\rho_\delta(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r| \le \delta \\
\delta(|r|-\frac{1}{2}\delta), & |r| > \delta
\end{cases}
\]

意义：

- 小误差时像 L2，正常拟合
- 大误差时像 L1，抑制 outlier

#### 3.2 Palm normal loss

\[
\mathcal{L}_{normal}
=
\|\mathbf{n}_{orca}(\mathbf{q}_t)-\mathbf{n}_{mp}(\mathbf{y}_t)\|_2^2
\]

意义：

- 不只拟合点位置
- 还要求掌面朝向一致

#### 3.3 Prior loss

把 `corrected` 当作启发式先验：

\[
\mathcal{L}_{prior}
=
\|\mathbf{q}_t - \tilde{\mathbf{q}}_t\|_2^2
\]

意义：

- `corrected` 提供合理初始化
- 优化不要偏离结构化几何估计太远

#### 3.4 Temporal loss

\[
\mathcal{L}_{temporal}
=
\|\mathbf{q}_t - \mathbf{q}_{t-1}^*\|_2^2
\]

意义：

- 抑制相邻帧乱跳
- 让状态变化更连续

#### 3.5 Acceleration loss

\[
\mathcal{L}_{acceleration}
=
\|\mathbf{q}_t - 2\mathbf{q}_{t-1}^* + \mathbf{q}_{t-2}^*\|_2^2
\]

意义：

- 惩罚一闪而过的瞬时跳变
- 抑制二阶时间不连续
- 非常适合对抗 MediaPipe 的单帧漂移和短时抖动

#### 3.6 Default pose loss

意义：

- 防止解跑到特别极端、特别不自然的位置
- 给优化一个温和稳定中心

#### 3.7 Boundary loss

意义：

- 防止 actuator 长期贴在边界
- 提高解的物理可行性

### 4. `corrected` 和 `optimized_action` 的根本区别

`corrected`：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

特点：

- 直接映射
- 速度快
- 结构化低维
- 但本质还是逐帧

`optimized_action`：

\[
\mathbf{q}_t^*
=
\arg\min
\mathcal{L}(\mathbf{q}_t,\mathbf{y}_t,\mathbf{q}_{t-1}^*,\mathbf{q}_{t-2}^*)
\]

特点：

- 优化求解
- 受 MuJoCo forward kinematics 约束
- 有 prior、法向、默认姿态、边界约束
- 有 temporal / acceleration 正则
- 对异常观测更鲁棒

因此：

- `corrected` 更像 **rule-based structured projection**
- `optimized_action` 更像 **physics-constrained robust temporal state estimation**

### 5. 为什么 `optimized_action` 往往比 `optimized_full` 更适合分类

`optimized_action` 是 17 维 latent actuator state：

\[
\mathbf{q}_t^* \in \mathbb{R}^{17}
\]

`optimized_full` 是再投影回 63 维 landmark 坐标：

\[
\mathbf{y}_t^* = h(\mathbf{q}_t^*)
\]

当前实验说明，对分类来说更有用的往往不是最完整的高维重建，而是：

- 低维
- 结构化
- 噪声更少
- few-shot 下更不容易过拟合

这也是为什么当前反复观察到：

\[
optimized\_action > optimized\_full
\]

### 6. 最简洁的表述方式

后续论文里最推荐这样解释：

- `corrected`：**a heuristic embodiment-aware projection into ORCA actuator space**
- `optimized_action`：**a MuJoCo-constrained robust temporally regularized latent hand state**

或者更口语一点：

- `corrected` 是“基于 ORCA 结构先验的启发式低维映射”
- `optimized_action` 是“基于 MuJoCo 约束优化的鲁棒时序潜在状态估计”
