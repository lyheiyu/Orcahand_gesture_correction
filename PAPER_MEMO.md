# 论文备忘录

## 暂定题目

**Robust Hand Landmark Denoising via MuJoCo-Constrained Temporal Optimization**

这个标题目前最稳，原因是：

- `Robust` 对应当前问题里的漂移、抖动和瞬时异常跳变
- `Hand Landmark Denoising` 明确了输入输出对象是 noisy hand landmarks，而不是泛化的 gesture semantics
- `MuJoCo-Constrained Temporal Optimization` 准确描述了方法核心：不是普通滤波，而是在 ORCA actuator latent space 中做受 MuJoCo forward kinematics 约束的时序优化

相比 `gesture denoising`，`landmark denoising` 更精确，也更贴近当前代码和实验。

## 摘要开头建议

下面这句可以直接作为英文摘要中的关键 framing：

**Unlike conventional temporal smoothing, the proposed denoising process is performed in an ORCA actuator space and constrained by MuJoCo forward kinematics, enabling physically plausible and temporally consistent refinement of noisy hand observations.**

如果想更完整一点，也可以用这一段作为摘要前两句草稿：

**Vision-based hand landmark estimators such as MediaPipe often suffer from frame-level jitter, drift, and transient outliers under challenging viewpoints and self-occlusion. To address this issue, we propose a MuJoCo-constrained temporal optimization framework that denoises noisy hand landmarks in an ORCA actuator space, producing physically plausible and temporally consistent latent hand trajectories.**

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
- `optimized_action_v1`
  - 初版 MuJoCo 优化得到的 ORCA latent actuator state，只有弱时间正则
- `optimized_action_v2`
  - 当前增强版 MuJoCo 优化得到的 ORCA latent actuator state，加入 Huber landmark loss 和 acceleration temporal regularization
- `optimized_full_v2`
  - 将 `optimized_action_v2` 再投影回 landmark 空间后得到的完整重建点坐标

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

### 2. `optimized_action_v1`

`optimized_action_v1` 是第一版 MuJoCo 优化得到的 ORCA latent actuator state。

它仍然是 17 维：

\[
\mathbf{q}_t^* \in \mathbb{R}^{17}
\]

但它不是直接由规则映射得到，而是通过优化求解：

\[
\mathbf{q}_t^*
=
\arg\min_{\mathbf{q}}
\mathcal{L}_{v1}(\mathbf{q})
\]

它与 `corrected` 的关系是：

\[
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

作为优化的初始化和先验：

\[
\mathcal{L}_{prior}
=
\|\mathbf{q}_t - \tilde{\mathbf{q}}_t\|_2^2
\]

第一版目标函数可以概括为：

\[
\mathcal{L}_{v1}(\mathbf{q}_t)
=
\lambda_l \mathcal{L}_{landmark}
+ \lambda_n \mathcal{L}_{normal}
+ \lambda_p \mathcal{L}_{prior}
+ \lambda_s \mathcal{L}_{temporal}
+ \lambda_d \mathcal{L}_{default}
+ \lambda_b \mathcal{L}_{boundary}
\]

其中 landmark loss 为：

\[
\mathcal{L}_{landmark}
=
\sum_i
\|h_i(\mathbf{q}_t) - \mathbf{y}_{t,i}\|_2^2
\]

palm normal loss 为：

\[
\mathcal{L}_{normal}
=
\|\mathbf{n}_{orca}(\mathbf{q}_t)
-
\mathbf{n}_{mp}(\mathbf{y}_t)\|_2^2
\]

temporal loss 为：

\[
\mathcal{L}_{temporal}
=
\|\mathbf{q}_t - \mathbf{q}_{t-1}^*\|_2^2
\]

所以 `optimized_action_v1` 的数学内涵是：

**带结构先验和弱时序正则的单帧逆运动学拟合。**

它比 `corrected` 更物理，因为它通过 MuJoCo forward kinematics 检查候选 actuator state 是否能解释观测关键点。但它的时序项只约束当前状态接近上一帧：

\[
\|\mathbf{q}_t - \mathbf{q}_{t-1}^*\|_2^2
\]

因此对一闪而过的 MediaPipe outlier 抵抗力有限。

### 3. `optimized_action_v2`

`optimized_action_v2` 是当前增强版表示，也是目前最有价值的表示。

它仍然输出 17 维 actuator latent state：

\[
\mathbf{q}_t^* \in \mathbb{R}^{17}
\]

但相比 v1，它新增了两个关键机制：

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

因此 `optimized_action_v2` 的数学内涵是：

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

`optimized_action_v2` 是：

\[
\mathbf{q}_t^*
=
\arg\min
\mathcal{L}(\mathbf{q}_t, \mathbf{y}_t, \mathbf{q}_{t-1}^*, \mathbf{q}_{t-2}^*)
\]

因此它更接近真正的 tracking / filtering 方法。

### 4. `optimized_full_v2`

`optimized_full_v2` 不是新的优化变量，而是把 `optimized_action_v2` 得到的 latent state 重新投影回 landmark 空间。

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

- `optimized_action_v2` 是 17 维 latent actuator state
- `optimized_full_v2` 是 63 维 reconstructed landmark representation

所以 `optimized_full_v2` 是一种：

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
| `optimized_action_v1` | 17 | MuJoCo 优化 | 是 | 弱时序 | 结构一致 latent state |
| `optimized_action_v2` | 17 | 鲁棒 MuJoCo 时序优化 | 是 | 较强时序 | 稳定且判别性强的 latent state |
| `optimized_full_v2` | 63 | v2 经 MuJoCo forward 重建 | 间接 | 继承 v2 | 结构一致重建 landmarks |

最核心的数学区别可以简写为：

\[
\textbf{corrected:}
\quad
\tilde{\mathbf{q}}_t = g(\mathbf{y}_t)
\]

\[
\textbf{optimized\_action\_v1:}
\quad
\mathbf{q}_t^*
=
\arg\min_{\mathbf{q}}
\left[
\|h(\mathbf{q})-\mathbf{y}_t\|^2
+
\lambda_s\|\mathbf{q}-\mathbf{q}_{t-1}^*\|^2
+
\lambda_p\|\mathbf{q}-\tilde{\mathbf{q}}_t\|^2
\right]
\]

\[
\textbf{optimized\_action\_v2:}
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
\textbf{optimized\_full\_v2:}
\quad
\mathbf{y}_t^*
=
h(\mathbf{q}_t^*)
\]

因此，论文里最推荐的解释是：

**`corrected` 是 heuristic embodiment-aware projection；`optimized_action_v1` 是 weak temporal MuJoCo fitting；`optimized_action_v2` 是 robust temporally regularized MuJoCo latent-state estimation；`optimized_full_v2` 是 optimized latent state 的 forward-kinematic landmark reconstruction。**

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

但当前结果显示，最适合分类的是 \(\mathbf{q}_t^*\) 这个 `optimized_action_v2` latent state，而不是重建后的 \(\mathbf{y}_t^*\)。

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
- `optimized_action_v2`
  - `velocity_mean = 0.334345`
  - `acceleration_mean = 0.357958`
- `optimized_full_v2`
  - `velocity_mean = 0.451672`
  - `acceleration_mean = 0.519420`

当前最重要的稳定性发现是：

**`optimized_action_v2` 同时具有最低的 velocity 和 acceleration 指标，说明加入 Huber loss 与 acceleration temporal regularization 后，MuJoCo-constrained latent actuator state 明显降低了时间抖动。**

### 2. Few-Shot Sequence Classification

当前 v2 数据集上的结果：

- `optimized_action_v2`: `0.8500 ± 0.1159`
- `corrected`: `0.8063 ± 0.1081`
- `optimized_full_v2`: `0.7125 ± 0.1858`

当前排序为：

\[
optimized\_action\_v2 > corrected > optimized\_full\_v2
\]

结合之前 raw baseline：

- `raw`: `0.6312 ± 0.1504`

因此当前总体趋势为：

\[
optimized\_action\_v2 > corrected > optimized\_full\_v2 > raw
\]

### 3. Multi-Classifier Comparison

后续补充的多分类器实验表明，当前观察并不是单一分类器偶然造成的。基于 `gesture_sequence_dataset_optimized_v2.csv`，`shots_per_class = 3`，`repeats = 20`：

#### SVM

- `raw`: `accuracy = 0.6313`, `macro_f1 = 0.6145`, `kappa = 0.4401`
- `corrected`: `accuracy = 0.8063`, `macro_f1 = 0.7808`, `kappa = 0.7025`
- `optimized_action_v2`: `accuracy = 0.8500`, `macro_f1 = 0.8290`, `kappa = 0.7674`
- `optimized_full_v2`: `accuracy = 0.7125`, `macro_f1 = 0.6723`, `kappa = 0.5675`

#### KNN

- `raw`: `accuracy = 0.5875`, `macro_f1 = 0.5392`, `kappa = 0.3595`
- `corrected`: `accuracy = 0.7875`, `macro_f1 = 0.7671`, `kappa = 0.6822`
- `optimized_action_v2`: `accuracy = 0.8063`, `macro_f1 = 0.7802`, `kappa = 0.7114`
- `optimized_full_v2`: `accuracy = 0.6375`, `macro_f1 = 0.6120`, `kappa = 0.4528`

#### RandomForest

- `raw`: `accuracy = 0.5938`, `macro_f1 = 0.5418`, `kappa = 0.3688`
- `corrected`: `accuracy = 0.8688`, `macro_f1 = 0.8679`, `kappa = 0.8030`
- `optimized_action_v2`: `accuracy = 0.8938`, `macro_f1 = 0.8834`, `kappa = 0.8395`
- `optimized_full_v2`: `accuracy = 0.7625`, `macro_f1 = 0.7480`, `kappa = 0.6371`

#### MLP

- `raw`: `accuracy = 0.6250`, `macro_f1 = 0.5934`, `kappa = 0.4188`
- `corrected`: `accuracy = 0.7625`, `macro_f1 = 0.7517`, `kappa = 0.6410`
- `optimized_action_v2`: `accuracy = 0.8125`, `macro_f1 = 0.8013`, `kappa = 0.7148`
- `optimized_full_v2`: `accuracy = 0.6875`, `macro_f1 = 0.6653`, `kappa = 0.5215`

当前最重要的新发现是：

- `optimized_action_v2` 在四个分类器中都保持强竞争力
- `RandomForest + optimized_action_v2` 是目前最强组合
- `corrected` 仍然稳定优于 `raw`
- `optimized_full_v2` 通常不如 `optimized_action_v2`，支持“低维结构化 latent representation 优于高维重建坐标”的论点

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
3. `optimized_action_v1`
   - 初版只有弱时间正则，因此虽然结构一致，但不足以稳定一闪而过的异常观测
4. `optimized_action_v2`
   - Huber loss 降低异常 landmark 的影响
   - acceleration loss 抑制二阶时间跳变
   - 因此同时获得更低 jitter 和更高分类准确率
5. `optimized_full_v2`
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
5. 得到的 `optimized_action_v2` 同时降低 jitter，并提升 sequence-level few-shot classification
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

## 下一步必须做的实验

### 1. PCA-17 Baseline

目的：

确认 `corrected` / `optimized_action_v2` 的优势不是单纯因为从 63 维降到了 17 维。

需要比较：

- `raw`
- `PCA(raw)-17`
- `corrected`
- `optimized_action_v2`

如果：

\[
optimized\_action\_v2 > corrected > PCA17 > raw
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
- `optimized_action_v1`
- `optimized_action_v2_no_huber`
- `optimized_action_v2_no_acceleration`
- `optimized_action_v2_full`

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
