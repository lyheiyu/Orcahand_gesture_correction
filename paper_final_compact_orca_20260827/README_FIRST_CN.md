# 先读这里

项目代码、数据、实验、论文修改和中断恢复的完整中文手册位于项目根目录：

```text
PROJECT_AND_PAPER_GUIDE_CN.md
```

最常用操作：

```powershell
# 阅读英文 Markdown 论文
code .\paper_final_compact_orca_20260827\paper_rewritten.md

# 修改图表后重新生成
python .\generate_final_compact_paper_assets.py

# 重新生成冻结 final test 的主文和补充 CM
python .\generate_frozen_confusion_matrices.py

# 生成每个 CM 单元格的 mean/std Excel 和稳定性图
C:\Users\31734\anaconda3\python.exe .\export_cm_stability_excel.py

# 查看冻结数字
code .\paper_final_compact_orca_20260827\FINAL_RESULTS_MASTER.md

# 投稿前逐项检查
code .\paper_final_compact_orca_20260827\SENSORS_SUBMISSION_CHECKLIST_CN.md
```

不要直接修改 `figures/` 中的 PNG 或 `tables/` 中的生成表。图片和表格的源代码是：

```text
generate_final_compact_paper_assets.py
```

正式正文源文件是 `paper_rewritten.tex`；便于阅读和逐句修改的版本是 `paper_rewritten.md`。目前生成器输出 9 张图片和 12 张 CSV/LaTeX 表格，其中表 10--12 分别记录优化器参数、分类器参数和受控扰动协议。
