# 论文备忘录

## 暂定题目

基于 ORCA 结构先验与 MuJoCo 约束优化的手势表示研究

## 当前核心问题

本项目研究的问题不是“机械手能不能模仿人手动作”，而是：

**MediaPipe 手部关键点存在抖动和几何不稳定时，什么样的结构化表示最适合 few-shot 手势识别？**

也就是说，我们关心的是：

- 原始视觉关键点是否足够好
- 引入 ORCA 手结构先验后是否能提升识别
- MuJoCo 优化出来的表示，是否真的比简单结构映射更适合分类

## 当前方法结构

目前已经有 5 类表示：

- `raw`
  - MediaPipe 归一化后的 `21x3` 关键点
- `geom`
  - 从关键点中提取的几何特征
- `corrected`
  - 基于 ORCA actuator 空间的低维结构校正特征
- `optimized_action`
  - 通过 MuJoCo 优化得到的 ORCA latent actuator state
- `optimized_full`
  - 将优化后的 ORCA state 再投影回 landmark 空间后得到的完整重建点坐标

## 方法理解

项目当前最重要的理论框架是：

1. MediaPipe landmarks 是带噪观测，不是真值
2. ORCA hand 提供结构先验和可行状态空间
3. MuJoCo 提供前向运动学约束
4. 手势分类最终依赖的是“对类别有判别力的表示”，而不一定是“几何上最完整的重建”

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

当前优化目标为：

\[
\mathbf{q}_t^* = \arg\min_{\mathbf{q} \in \mathcal{Q}}
\lambda_l \mathcal{L}_{landmark}
+ \lambda_n \mathcal{L}_{normal}
+ \lambda_p \mathcal{L}_{prior}
+ \lambda_s \mathcal{L}_{temporal}
+ \lambda_d \mathcal{L}_{default}
+ \lambda_b \mathcal{L}_{boundary}
\]

其中：

- `landmark loss`
  - 拟合稀疏关键点
- `normal loss`
  - 拟合掌面法向
- `prior loss`
  - 约束解不要偏离启发式 ORCA 投影太远
- `temporal loss`
  - 约束相邻帧平滑
- `default loss`
  - 防止过度偏离默认姿态
- `boundary loss`
  - 防止贴住 actuator 边界

最终重建后的优化点为：

\[
\mathbf{y}_t^* = h(\mathbf{q}_t^*)
\]

## 当前最重要的实验设置

目前实验是：

- sequence-level classification
- few-shot setting
- `shots_per_class = 3`
- `repeats = 20`
- 分类器：`SVM`
- 类别：`6`, `7`, `8`

当前数据库已包含：

- 原始 `raw_*`
- `geom_*`
- `corrected_*`
- `optimized_action_*`
- `optimized_sparse_*`
- `optimized_full_*`
- `optimized_loss_*`

## 当前关键实验结果

在当前扩充后的 sequence 数据集上：

- `corrected`: `0.8063 ± 0.1081`
- `optimized_action`: `0.6687 ± 0.1267`
- `raw`: `0.6312 ± 0.1504`
- `optimized_full`: `0.6250 ± 0.1677`

当前排序为：

\[
corrected > optimized\_action > raw \approx optimized\_full
\]

## 当前最重要的发现

当前最重要、最有意思的研究发现是：

**最适合 few-shot sequence 手势分类的表示，不是原始 MediaPipe landmarks，也不是 MuJoCo 优化后重建的完整 landmarks，而是一个低维、结构化、任务导向的 ORCA-aware corrected actuator representation。**

这个发现是有研究价值的，因为它说明：

- 结构约束是有效的
- 但“更完整的几何重建”不等于“更好的分类特征”
- reconstruction-oriented representation 和 discrimination-oriented representation 不一定一致

## 当前现象的原理解释

目前最合理的解释是：

1. `raw`
   - 信息多，但噪声也大，few-shot 下容易被噪声拖累
2. `corrected`
   - 通过结构约束把高维 noisy landmark 压缩成低维、语义明确、关节一致的表示，因此信噪比更高，更适合分类
3. `optimized_action`
   - 是一个结构一致的 latent state，但优化目标更偏向几何拟合与平滑，而不是最大化分类判别性，所以没有超过 `corrected`
4. `optimized_full`
   - 是把 latent state 又还原回高维点坐标，维度升高了，few-shot 下学习更难，而且当前重建点更偏向“结构一致”而不是“分类判别强”

## 当前适合写进论文的安全结论

目前最安全的结论是：

**ORCA-aware low-dimensional corrected features outperform both raw MediaPipe landmarks and the current MuJoCo-optimized landmark reconstruction in few-shot sequence-level gesture classification.**

中文版可以写成：

**在当前 few-shot sequence 手势识别任务中，基于 ORCA 结构先验的低维 corrected 表示优于原始 MediaPipe landmarks，也优于当前 MuJoCo 优化后重建的 landmark 表示。**

## 当前不适合夸大的结论

目前不要写成：

- MuJoCo optimized landmarks improve classification
- full physics-based reconstruction
- exact hand pose recovery
- optimized representation is the best representation

这些都和当前结果不一致。

## 当前论文最好的主线

当前最稳的主线是：

1. MediaPipe landmarks 有噪声和抖动
2. 引入 ORCA 结构先验后，低维 corrected 表示显著提升 few-shot sequence 分类效果
3. MuJoCo 优化提供了一个结构一致的 refinement framework
4. 但当前最适合分类的仍然是 corrected actuator-space representation，而不是优化后高维重建点

## 为什么这个结果有意思

这个结果有意思，不是因为“优化失败了”，而是因为它揭示了一个更深的事实：

**对分类任务来说，最优表示未必是最原始的，也未必是最完整几何重建的，而可能是一个结构约束下的低维中间表示。**

这实际上是一个关于：

- representation design
- embodiment prior
- few-shot robustness
- reconstruction vs discrimination tradeoff

的研究问题。

## 当前项目所处阶段

当前状态适合定义为：

- 已经有一个清晰的研究故事
- 已经有初步可重复的实验结果
- 已经可以开始写论文 draft
- 但还不适合做特别强的最终定论

更像：

- workshop paper
- short paper
- early-stage conference submission
- 或完整论文的第一版草稿

## 当前不足

目前还存在这些不足：

- sequence 数量还不够大
- 类别数还比较少
- 目前主要只跑了 `SVM`
- 优化仍然是 sparse correspondence fitting
- 还没有直接评价 jitter reduction 指标
- 还没有专门做 optimization loss 的 ablation

## 下一步建议

最推荐的下一步有 5 个：

1. 继续补 sequence 数据
2. 输出 `macro-F1 mean/std`
3. 增加更多分类器比较
4. 增加 jitter / smoothness 指标
5. 做一个专门的 ablation，分析为什么 optimized 不如 corrected

## 一句话版本

当前项目最值得记录的结论是：

**结构先验是有效的，但在 few-shot 手势识别中，最优表示是低维的 ORCA-aware corrected actuator-space representation，而不是优化后重建的高维 landmarks。**
