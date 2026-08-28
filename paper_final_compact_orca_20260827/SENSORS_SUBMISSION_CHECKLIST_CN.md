# Sensors 投稿收尾清单

这份清单用于把当前科研稿件变成可投稿稿件。带 `[TODO]` 的信息不能从代码或 CSV 可靠推断，必须依据真实采集记录填写。

## 一、投稿前必须填写

- [ ] 作者单位、机构邮箱、通讯作者。
- [ ] 参与者人数，以及每位参与者的 sequence 数量。
- [ ] 每位参与者的录制 session 数量和日期范围。
- [ ] 六类手势各自的 sequence 数量。
- [ ] 相机或摄像设备型号、分辨率、帧率。
- [ ] 相机距离、背景、照明和采集姿态约定。
- [ ] MediaPipe 失败帧、低置信度帧和人工筛选的处理规则。
- [ ] 伦理审批编号，或真实有效的豁免说明。
- [ ] 参与者知情同意和可识别图片/视频发表许可。
- [ ] Data Availability、Funding、Author Contributions、Conflicts of Interest。
- [ ] ORCA 模型来源和 JointAngle-11 的准确文献条目。

## 二、已经从代码补齐

- [x] 17 个 actuator 的名称、语义和弧度上下限。
- [x] MediaPipe 几何量到 actuator 的角度、flexion、spread、wrist 和 thumb 映射。
- [x] 七项损失的公式以及固定权重。
- [x] Huber 阈值 `0.08`。
- [x] SciPy L-BFGS-B、投影初始化、每帧最多 `120` 次迭代。
- [x] 第一帧、第二帧和 sequence history reset 的处理。
- [x] 右手模型、wrist origin、尺度归一化和未做通用左右手 canonicalization 的范围说明。
- [x] SVM、KNN、RandomForest 和 MLP 的冻结参数。
- [x] Compact Refined-7 的 `D/W/I/R/S` 定义、候选 K、one-standard-error rule 和误差线含义。
- [x] Gaussian、spike、dropout 的幅度、持续时间、受影响 landmarks 和随机种子协议。
- [x] actuator perturbation MAE 的数学定义。
- [x] Holm correction family：每个 classifier-metric 下的 6 个预定义比较。
- [x] 20 次重复的置信区间只反映 few-shot 训练样本选择变化。

## 三、Word 排版检查

- [ ] 不直接把 Markdown 文本粘贴为投稿 Word；使用 Pandoc 或 Word 模板样式转换。
- [ ] 清除正文中的 `**...**`、反引号、`->` 和代码形式公式。
- [ ] 所有公式使用 Word Equation 或 LaTeX 正式公式。
- [ ] Figure 和 Table 使用 Word Caption 自动编号，不手工输入编号。
- [ ] 所有正文引用使用 Cross-reference，避免图表移动后编号失效。
- [ ] 图注紧跟对应图片，表题位于表格上方。
- [ ] 检查页眉、页脚、页码和图注是否越界或被裁切。
- [ ] Limitations 从 1 开始连续编号。
- [ ] 删除转换过程中出现的孤立 `text` 或乱码字符。
- [ ] References 必须实际插入，不保留“从 references.bib 插入”的说明文字。

## 四、术语冻结

论文正文只使用以下名称：

- `Actuator Projection-17`：代码 legacy key 为 `corrected`。
- `Refined ORCA-17`：代码 key 为 `optimized_action`。
- `Compact Projection-7`：投影状态的冻结七维读取。
- `Compact Refined-7`：优化状态的冻结七维读取。
- `Reconstructed ORCA Landmarks`：代码 key 为 `optimized_full`。
- `JointAngle-11`：外部 11 维几何基线。
- `Projection PCA-11` / `Refined PCA-11`：训练集拟合的维度控制基线。

不要在论文自然语言中继续混用 `Corrected`、`Optimized Action`、`OA-7`、`Compact ORCA-7` 或 `Optimized Full`。代码和 CSV 字段不需要因此重命名。

## 五、最终编译检查

- [ ] 删除旧的 LaTeX 辅助文件后完整执行 LaTeX -> BibTeX -> LaTeX -> LaTeX。
- [ ] PDF 中不存在 `[?]`、`??` 或缺失 bibliography。
- [ ] 所有图表按首次出现顺序编号。
- [ ] Figure 6 图注明确误差线为 `95% CI`。
- [ ] spike reduction 全文统一为 `17.7%`。
- [ ] 标题、摘要、贡献、RQ、结果和结论使用同一套术语。
- [ ] 最终结论不声称真实 3D ground-truth recovery、subject-independent generalization 或普遍优于所有分类器。

## 六、当前可投稿边界

现有结果可以支持：ORCA actuator projection 提供紧凑语义表示；MuJoCo 因果细化降低同一 actuator 空间中的时间变化和受控扰动敏感度；Compact Refined-7 在冻结 final test 上对 SVM/KNN 显著优于 JointAngle-11。

现有结果不能支持：准确恢复真实人体 3D 手姿、跨参与者泛化、完全解决真实遮挡，或 Refined ORCA 对所有分类器都最好。
