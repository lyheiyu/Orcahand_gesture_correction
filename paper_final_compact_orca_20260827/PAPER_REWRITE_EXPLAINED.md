# 这次论文重写做了什么

## 核心变化

这次不再把论文写成“Optimized Action 必须在所有分类器上超过 Corrected”。那种主线既不符合全部结果，也会让论文容易被反例推翻。

现在论文分成三个相互独立但连续的贡献：

1. **Actuator Projection-17**：把 63 维 MediaPipe 点变成 17 维 ORCA actuator state。它对应代码中的 `corrected`，作用是结构化、语义化和初始化。
2. **MuJoCo-constrained Temporal Refinement**：用 Huber、前向运动学、actuator bounds、一阶和二阶时序项优化轨迹。它的主要证据是更稳定、对受控扰动更不敏感，而不是每个分类器都必须更高。
3. **Compact Refined-7**：只在 development set 上选择 7 个语义 actuator，然后冻结，在 final test 上评价。它解决 17 维 refined state 可能包含冗余的问题。

## 为什么这样更可信

- `corrected` 分类强并不会否定论文，因为它证明 ORCA structured representation 本身有价值。
- `optimized_action` 的价值由同空间稳定性和扰动实验直接证明，不再只依赖分类准确率。
- Compact Refined-7 在 SVM 和 KNN 上显著优于 JointAngle-11；RF 和 MLP 不显著也被如实保留。
- 选择 Compact7 与最终测试严格分开，避免使用 final labels 反复挑方法。

## 最安全的一句话结论

ORCA actuator projection 提供了紧凑、可解释的手部表示；MuJoCo causal refinement 降低了 actuator trajectory 的时间变化和受控输入扰动敏感性；development-only 选择的 Compact Refined-7 在更少特征下，对 SVM 和 KNN 的 frozen final test 显著优于 JointAngle-11，但这种优势不是所有分类器都显著。

## 不再使用的旧主线

- 不再说 `optimized_action` 在所有条件下都是最好的分类表示。
- 不再把 `corrected` 写成与优化方法竞争的“最终算法”。
- 不再把 39/56 sequence 的早期结果作为主实验。
- 不再把 mean/std/max/delta 作为主要时序协议。
- 不再跨 landmark space 和 actuator space 直接比较 jitter 数值。
- 不再把 synthetic corruption 称为 physical recovery 或 ground-truth correction。

## 最终文件

- `paper_rewritten.md`：最容易阅读和继续修改的完整英文稿。
- `paper_rewritten.tex`：包含全部最终图表的 LaTeX 稿。
- `FINAL_RESULTS_MASTER.md`：冻结数字总账。
- `FINAL_CLAIMS_AND_LIMITATIONS.md`：可说和不可说的边界。
- `FINAL_EXPERIMENT_MAP.md`：每个 research question 对应哪项实验。
- `FINAL_FIGURE_INDEX.md`：图片用途和重生成方式。
- `figures/`：9 张最终图片。
- `tables/`：9 组 CSV 和 LaTeX 表格。
