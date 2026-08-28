# ORCA 项目与论文完整操作手册

最后更新：2026-08-27

这份文件是本项目的总入口。以后聊天中断、Codex 重置或换电脑后，先打开本文件，不需要依赖聊天记录。

---

## 1. 现在项目做到哪里了

当前论文科学主线已经冻结：

```text
MediaPipe 21x3 landmarks
-> Frame-wise ORCA actuator projection (17D)
-> MuJoCo-constrained causal temporal refinement (17D)
-> Frozen Compact ORCA-7
-> Resample16 sequence encoding
-> SVM / KNN / RandomForest / MLP
```

当前论文不再要求 `Refined ORCA-17` 在每个分类器上都超过代码中的 `corrected`。三个阶段分别承担不同作用：

- Frame-wise Actuator Projection：结构化和初始化，对应代码列名 `corrected_*`。
- Refined ORCA-17：降低 actuator trajectory 的时间变化和扰动敏感性。
- Compact Refined-7：删除冗余 actuator，形成最终紧凑分类表示。

最终结果使用：

- 571 条 sequence，26,260 帧，6 个中国舞手势类别。
- development/final = 456/115。
- 3-shot，20 repeats。
- Compact7 indices = `[3, 6, 9, 11, 12, 15, 16]`。
- Primary encoder = Resample16。

---

## 2. 最重要的文件在哪里

### 2.1 最终论文文件夹

```text
paper_final_compact_orca_20260827/
```

里面的主要文件：

| 文件 | 用途 | 是否可以直接修改 |
|---|---|---|
| `paper_rewritten.md` | 完整英文论文，最适合阅读和改句子 | 可以 |
| `paper_rewritten.tex` | LaTeX 投稿稿，包含图表引用 | 可以 |
| `references.bib` | 参考文献 | 可以添加，不要随意删已引用 key |
| `FINAL_RESULTS_MASTER.md` | 最终数字总账 | 除非重新完成冻结实验，否则不要改数字 |
| `FINAL_CLAIMS_AND_LIMITATIONS.md` | 可以说与不可以说的结论 | 建议先读再改摘要 |
| `FINAL_EXPERIMENT_MAP.md` | Research Question 与实验的对应关系 | 可以补说明 |
| `FINAL_FIGURE_INDEX.md` | 9 张图的用途和规则 | 可以补说明 |
| `PAPER_REWRITE_EXPLAINED.md` | 中文解释论文为什么这样重写 | 可以 |
| `figures/` | 最终 9 张论文图 | 不要手工覆盖，使用生成脚本 |
| `tables/` | 最终 CSV 和 LaTeX 表 | 不要手工改数字，使用生成脚本 |

### 2.2 最终出图程序

```text
generate_final_compact_paper_assets.py
```

这是修改论文图片最重要的程序。它只读取已经冻结的实验 CSV，不重新训练模型。

关键位置：

| 代码位置 | 作用 |
|---|---|
| 第 19-22 行附近 | 输入结果目录和最终输出目录 |
| `PRIMARY_REPS` | 主表显示哪些 representation |
| `DISPLAY` | 图例和表格显示名称 |
| `COLORS` | 每种 representation 的颜色 |
| `FROZEN` | Compact7 actuator indices |
| `figure_pipeline()` | Figure 1 方法流程图 |
| `table_actuators()` | Table 1-2 actuator 表 |
| `figure_actuators()` | Figure 2 actuator inventory |
| `figure_trajectory()` | Figure 3 代表性轨迹 |
| `table_and_figure_stability()` | Figure 4 和 stability 表 |
| `final_results()` | 最终分类表和 paired statistics 表 |
| `figure_dimension_control()` | Figure 5 |
| `figure_development_k()` | Figure 6 |
| `figure_final_joint()` | Figure 7 |
| `figure_dimension_efficiency()` | Figure 8 |
| `table_and_figure_perturbation()` | Figure 9 |
| `remaining_tables()` | Ablation 和 runtime 表 |

重新生成全部图片和表格：

```powershell
conda activate orca
python .\generate_final_compact_paper_assets.py
```

正常输出应为：

```text
output=...\paper_final_compact_orca_20260827
figures=9 tables=9
```

如果只想修改图片标题、颜色、字体、图例位置，可以编辑这个脚本后重新运行。不要直接在 PNG 上写字。

### 2.3 冻结实验结果

```text
diagnostics/orca_compact_selection_20260827/
```

重要源文件：

| 文件 | 内容 |
|---|---|
| `development_sequences.csv` | development sequence IDs |
| `final_test_sequences.csv` | final-test sequence IDs |
| `FINAL_COMPACT_ORCA_SPEC.md` | 冻结 Compact7 规格 |
| `compact_dimension_selection.csv` | development-only K 扫描 |
| `orca_actuator_inventory.csv` | 17 个 actuator 定义 |
| `final_test_results.csv` | 最终 mean/std/CI |
| `final_test_per_repeat.csv` | 每次 repeat 的原始结果 |
| `final_test_paired_comparisons_holm.csv` | Wilcoxon、Holm p、effect size |

这些是最终论文结果的 source of truth。论文中的数字应该从这里读取，不应凭记忆手写。

---

## 3. 核心算法代码怎么理解

### 3.1 MediaPipe landmarks 转 17D actuator

文件：

```text
src/orca_sim/gesture_features.py
```

阅读顺序：

1. `normalize_landmarks()`：以 wrist 为原点并按掌宽/掌长缩放。
2. `palm_normal_vector()`：计算统一方向的 palm normal。
3. `extract_hand_features()`：计算 wrist、finger flexion、abduction 和 thumb 特征。
4. `OrcaFeatureProjector.corrected_vector()`：把手部几何特征映射到 17 个 ORCA actuator ranges。
5. `_set_signed()`、`_set_signed_neutral()`、`_set_unit()`：完成 feature 到 actuator range 的缩放。

17 个 actuator 的实际赋值集中在 `corrected_vector()` 中，大约在第 284-300 行：

```python
self._set_unit(action, "right_i-mcp_actuator", features.index_mcp)
self._set_unit(action, "right_i-pip_actuator", features.index_pip)
```

如果某根手指方向相反，先检查这里的符号和对应 feature，不要先修改 MuJoCo loss。

论文中的名称：

```text
Actuator Projection-17
```

CSV 和旧代码中的名称：

```text
corrected_0 ... corrected_16
```

### 3.2 MuJoCo 优化

文件：

```text
src/orca_sim/mujoco_optimizer.py
```

最重要的类：

```python
MujocoHandPoseOptimizer
```

最重要的位置：

| 代码 | 作用 |
|---|---|
| `OptimizationWeights` | 所有 loss weights 和 Huber delta |
| `MujocoHandPoseOptimizer.__init__()` | 载入右手 MuJoCo model 和 actuator bounds |
| `optimize()` | 每帧优化入口 |
| 内部 `loss_terms()` | 计算所有未加权 loss 和 weighted total |
| `_landmark_loss()` | component-wise Huber landmark loss |
| `_forward_sparse_points()` | actuator -> MuJoCo forward kinematics -> sparse points |
| `_normalize_model_points()` | MuJoCo points 的 wrist/scale normalization 和 palm normal |

默认权重当前为：

```python
landmark = 1.0
palm = 0.2
prior = 0.3
temporal = 0.1
acceleration = 0.15
default_pose = 0.15
boundary = 0.05
huber_delta = 0.08
```

论文已经冻结。除非明确开始新实验，不要修改这些参数后继续使用旧结果。

`optimize()` 的三个关键输入：

```python
initial_action      # 当前帧 frame-wise actuator projection
prev_action         # 上一帧 optimized action
prev_prev_action    # 上上帧 optimized action
```

因此它不是独立单帧 correction，而是 causal frame-wise temporal refinement。

### 3.3 Palm normal 修复

统一 convention 位于：

```text
src/orca_sim/gesture_features.py::palm_normal_vector()
src/orca_sim/mujoco_optimizer.py::_normalize_model_points()
```

两处都使用：

```python
np.cross(palm_forward, palm_across)
```

不要只修改其中一处。对应 regression tests 位于：

```text
tests/test_mujoco_optimizer.py
```

运行：

```powershell
python -m pytest .\tests\test_mujoco_optimizer.py -q
```

### 3.4 Refined ORCA-17 和 Reconstructed ORCA Landmarks

生成逻辑位于：

```text
augment_dataset_with_optimization.py
```

输出列：

```text
optimized_action_0 ... optimized_action_16
optimized_sparse_0 ... optimized_sparse_23
optimized_full_0 ... optimized_full_62
optimized_loss_*
```

`optimized_action` 是主要 actuator latent state。

`optimized_full` 是把 actuator state 通过 MuJoCo forward kinematics 投影回 63D point representation。它主要用于 landmark-space visualization，不是当前最佳分类 representation。

---

## 4. 数据采集、优化和合并

### 4.1 采集新的 sequence

通用命令：

```powershell
python .\collect_gesture_dataset.py `
  --label orchid_palm `
  --output .\gesture_sequence_dataset_more.csv `
  --hand-landmarker-model ".\hand_landmarker.task" `
  --target-hand right `
  --sequence-mode `
  --export-optimized `
  --version v2
```

采集窗口中的具体开始/停止按键以程序界面提示为准。每次开始录制会生成新的 `sequence_id`。

推荐标签：

```text
orchid_palm
orchid_finger
flower_pinch
prayer_beads
three_finger_bent
deer_horn
```

### 4.2 已经有 raw CSV，补 optimized columns

```powershell
python .\augment_dataset_with_optimization.py `
  --input .\new_raw_dataset.csv `
  --output .\new_optimized_dataset.csv `
  --version v2
```

这个程序会按 sequence 顺序传递 `prev_action`，并在新 sequence 开始时重置 history。

### 4.3 合并 CSV

只有 header 完全一致的 CSV 才能直接合并。

```powershell
python .\merge_gesture_datasets.py `
  --master .\gesture_sequence_dataset_master.csv `
  --sources .\new_optimized_dataset.csv
```

默认行为：

- 按 `label + sequence_id + frame_id` 去重。
- 合并前创建 `.bak` 备份。
- 不删除 source 文件。

如果出现：

```text
Header mismatch between master/source CSVs
```

不要强行合并。先确认两边是否都包含相同版本的 `raw_*`、`corrected_*` 和 `optimized_*` columns。必要时先用 `augment_dataset_with_optimization.py` 从共同的 raw schema 重新生成。

### 4.4 检查数据库类别和 sequence 数量

```powershell
Import-Csv .\gesture_sequence_dataset_master.csv |
  Group-Object label |
  Select-Object Name,Count
```

查看每类独立 sequence 数：

```powershell
Import-Csv .\gesture_sequence_dataset_master.csv |
  Group-Object label |
  ForEach-Object {
    [PSCustomObject]@{
      Label = $_.Name
      Sequences = ($_.Group.sequence_id | Sort-Object -Unique).Count
      Frames = $_.Count
    }
  }
```

---

## 5. 分类和稳定性实验

### 5.1 快速测试一个 representation

```powershell
python .\train_svm.py `
  --dataset .\your_dataset.csv `
  --feature-set optimized_action `
  --sequence-mode `
  --shots-per-class 3 `
  --repeats 20 `
  --classifier svm `
  --plot-confusion .\figures\cm_svm_oa.png `
  --results-csv .\figures\results.csv
```

替换 classifier：

```text
svm
knn
rf
mlp
```

注意：`train_svm.py --sequence-mode` 的旧默认聚合仍是 mean/std/max/delta。最终 Compact ORCA 论文的 primary result 使用的是 `run_compact_orca_selection.py` 中的 Resample16，不要把两套协议混为一谈。

### 5.2 Jitter evaluation

```powershell
python .\evaluate_jitter.py `
  --dataset .\your_dataset.csv `
  --feature-sets corrected optimized_action `
  --results-csv .\figures\jitter_actuator.csv `
  --plot .\figures\jitter_actuator.png
```

只在相同空间内比较：

- actuator space：`corrected` vs `optimized_action`。
- landmark space：`raw` vs smoothing vs `optimized_full`。

不要直接说 17D actuator velocity 比 63D raw landmark velocity 更低。

### 5.3 Compact ORCA 选择程序

文件：

```text
run_compact_orca_selection.py
```

主要函数：

- `freeze_outer_split()`：冻结 development/final split。
- `actuator_scores()`：计算 actuator utility。
- `select_semantic_subset()`：加入 finger semantic coverage。
- `development_validation()`：只在 development 内选择 K。
- `select_k()`：冻结 K。
- `evaluate_frozen_final()`：使用已经冻结的 specification 评估 final test。

原始两阶段命令是：

```powershell
python .\run_compact_orca_selection.py `
  --dataset .\your_dataset.csv `
  --output-dir .\diagnostics\new_compact_experiment `
  --stage select `
  --shot 3 `
  --repeats 20
```

选择和 specification 检查完以后才能运行：

```powershell
python .\run_compact_orca_selection.py `
  --dataset .\your_dataset.csv `
  --output-dir .\diagnostics\new_compact_experiment `
  --stage final `
  --shot 3 `
  --repeats 20
```

重要：当前 `diagnostics/orca_compact_selection_20260827` 已经完成 final test。不要为了得到更漂亮的结果重跑 selection 或改 Compact7，这会破坏冻结实验逻辑。如果有全新多参与者数据，应建立新的日期目录并称为 external validation。

---

## 6. 如何修改论文

### 6.1 只修改英语句子

先改：

```text
paper_final_compact_orca_20260827/paper_rewritten.md
```

确认意思后，再同步修改：

```text
paper_final_compact_orca_20260827/paper_rewritten.tex
```

Markdown 是容易阅读的版本，LaTeX 是最终投稿版本。两者不是自动同步的，因此改完要搜索相同句子并手动同步。

### 6.2 修改数字

先检查：

```text
paper_final_compact_orca_20260827/FINAL_RESULTS_MASTER.md
```

再检查对应 source CSV。不要只改正文数字而不改表格，也不要用旧实验数字替换冻结结果。

### 6.3 修改图片

编辑：

```text
generate_final_compact_paper_assets.py
```

然后运行：

```powershell
python .\generate_final_compact_paper_assets.py
```

### 6.4 修改表格显示

表格源也由 `generate_final_compact_paper_assets.py` 生成。常见修改：

- `DISPLAY`：名称。
- `PRIMARY_REPS`：主表行。
- `final_results()`：显示哪些 metrics。
- `write_latex_table()`：LaTeX 表格格式。

不要直接修改 `tables/table_04_final_classification.tex`，因为下次运行生成脚本会覆盖它。

### 6.5 添加参考文献

编辑：

```text
paper_final_compact_orca_20260827/references.bib
```

正文引用格式：

```latex
\cite{bib_key}
```

必须确保 `bib_key` 与 `.bib` 中完全一致。JointAngle-11 来源论文的精确 BibTeX 仍需要最终确认。

### 6.6 编译 LaTeX

当前电脑没有 `pdflatex` 和 `latexmk`，VS Code 的 LaTeX Workshop 插件本身不包含 TeX 编译器。

安装 MiKTeX 或 TeX Live 后，重新打开 VS Code，再检查：

```powershell
pdflatex -v
latexmk -v
```

进入最终论文目录：

```powershell
cd .\paper_final_compact_orca_20260827
latexmk -pdf .\paper_rewritten.tex
```

如果引用显示 `?`，清理辅助文件后完整运行 LaTeX/BibTeX 流程，或直接让 `latexmk` 处理。

---

## 7. 哪些内容已经冻结，不能随意改

以下内容属于 frozen scientific story：

- development/final sequence IDs。
- outer seed 和 split。
- Compact7 indices。
- 3-shot、20 repeats primary protocol。
- Resample16 primary encoder。
- 四个 classifier 的 frozen settings。
- final-test results 和 Holm-adjusted tests。
- optimizer loss weights。

可以修改：

- 论文语言和段落顺序。
- 图片颜色、字体和排版。
- 表格中指标的显示精度。
- Discussion、Limitations 和 future work。
- 新增真正独立的新数据 external validation。

如果要改变算法、loss weight、Compact indices 或 classifier hyperparameters，应创建新的实验目录，不能覆盖当前结果。

---

## 8. 中断或重置后怎么恢复

每次恢复只做以下步骤：

### Step 1：确认工作目录

```powershell
Get-Location
```

应该是：

```text
C:\D\projects\Orca robot hand\orca sim\orca_sim
```

### Step 2：打开本手册

```powershell
code .\PROJECT_AND_PAPER_GUIDE_CN.md
```

### Step 3：检查 Git 状态

```powershell
git status --short
```

不要使用 `git reset --hard`，否则可能删除未提交论文。

### Step 4：检查最终文件

```powershell
Get-ChildItem .\paper_final_compact_orca_20260827
Get-ChildItem .\paper_final_compact_orca_20260827\figures
```

### Step 5：检查生成脚本

```powershell
python -m py_compile .\generate_final_compact_paper_assets.py
```

### Step 6：需要时重新生成图表

```powershell
python .\generate_final_compact_paper_assets.py
```

这一步只重新读取冻结 CSV，不会重新训练。

---

## 9. 建议的 Git 保存方式

完成一个小阶段就保存一次：

```powershell
git status --short
git add .\PROJECT_AND_PAPER_GUIDE_CN.md
git add .\generate_final_compact_paper_assets.py
git add .\paper_final_compact_orca_20260827
git commit -m "Freeze compact ORCA paper package and workflow guide"
git push
```

如果数据文件超过 GitHub 100 MB，不要普通 `git add`。选择：

- Git LFS；
- Zenodo/OSF/Google Drive 保存数据，仓库只放下载说明和 checksum；
- 发布经过压缩或抽样的 reproducibility subset。

最终 CSV 结果和论文图通常很小，可以正常提交。

---

## 10. 常见错误

### `Header mismatch`

原因：两个 CSV schema 不同。先统一 raw/corrected/optimized columns，再合并。

### `UndefinedMetricWarning`

原因：某次 few-shot split 中某个类别没有预测样本。它不等于程序错误，但应使用 aggregate mean/std、macro-F1 和 common splits，不要只解释最后一次 report。

### `spawn latexmk ENOENT`

原因：只安装了 VS Code 插件，没有安装 TeX distribution，或 PATH 未刷新。

### 图没有更新

确认你运行的是：

```powershell
python .\generate_final_compact_paper_assets.py
```

并检查输出目录不是旧的 `figures/paper_rewrite_main`，而是：

```text
paper_final_compact_orca_20260827/figures
```

### 结果突然变化

检查：

- dataset 是否相同；
- sequence labels 是否相同；
- split manifest 是否相同；
- shot/repeats 是否相同；
- encoder 是 Resample16 还是 mean/std/max/delta；
- 是否重新拟合或改了 optimizer parameters。

---

## 11. 最终安全结论

可以写：

> ORCA actuator projection provides a compact semantic representation of MediaPipe landmarks. MuJoCo-constrained causal refinement reduces actuator-space temporal variation and sensitivity to controlled landmark corruption. A seven-actuator subset frozen on development data improves final-test performance over JointAngle-11 for SVM and KNN, while RandomForest and MLP differences are not significant after Holm correction.

不要写：

- exact 3D human pose recovery；
- MuJoCo always outperforms every baseline；
- full physics-based dynamics reconstruction；
- subject-independent generalization；
- real-time on all hardware；
- synthetic corruption equals real occlusion ground truth。

---

## 12. 最短操作清单

以后如果只记得三件事：

1. 改论文：`paper_final_compact_orca_20260827/paper_rewritten.md` 和 `.tex`。
2. 改图：`generate_final_compact_paper_assets.py`，然后重新运行它。
3. 查数字：`paper_final_compact_orca_20260827/FINAL_RESULTS_MASTER.md` 和 `diagnostics/orca_compact_selection_20260827/*.csv`。
