# LaTeX 报告使用说明

## 文件说明

- `report.tex`: 主 LaTeX 报告文件
- `Makefile`: 编译脚本（可选）
- `REPORT_README.md`: 本说明文件

## 编译方法

### 方法 1: 使用 Makefile（推荐）

```bash
# 编译 PDF
make

# 清理临时文件
make clean

# 完全清理（包括 PDF）
make cleanall

# 编译并查看
make view
```

### 方法 2: 直接使用 pdflatex

```bash
# 编译（需要运行两次以解决引用）
pdflatex report.tex
pdflatex report.tex

# 如果使用参考文献，还需要运行
bibtex report
pdflatex report.tex
pdflatex report.tex
```

### 方法 3: 使用在线 LaTeX 编辑器

推荐使用 Overleaf (https://www.overleaf.com/)：
1. 上传 `report.tex` 文件
2. 上传图片文件（在 `outputs/figures/` 目录下）
3. 在线编译和编辑

## 报告结构

报告包含以下章节：

1. **Introduction** - 引言和问题陈述
2. **Related Work** - 相关工作
3. **Methodology** - 方法论
   - Dataset（数据集）
   - Models（模型）
   - Evaluation Metrics（评估指标）
4. **Experimental Setup** - 实验设置
5. **Results** - 结果分析
   - Overall Performance（整体性能）
   - Performance by Model（各模型性能）
   - Detailed Results by Horizon（各时间范围详细结果）
   - ROC and Precision-Recall Curves
   - Confusion Matrices
   - Feature Importance Analysis
   - Class Imbalance Analysis
6. **Discussion** - 讨论
7. **Conclusion** - 结论
8. **References** - 参考文献
9. **Appendix** - 附录

## 需要修改的内容

### 1. 作者信息

在 `report.tex` 文件中找到：
```latex
\author{Your Name}
```
替换为你的名字。

### 2. 图片路径

确保图片文件路径正确。当前路径设置为：
```latex
\includegraphics[width=0.9\textwidth]{outputs/figures/performance/CO_model_comparison.png}
```

如果图片在不同位置，请修改路径。

### 3. 参考文献

在 `\begin{thebibliography}` 部分添加实际的参考文献。格式示例：
```latex
\bibitem{ref1}
Author A, Author B. (Year). Title. \textit{Journal Name}, Volume, Pages.
```

### 4. 结果数据

报告中的数值来自你的实验结果。如果需要更新：
- 检查 `outputs/results/classification_results_all_pollutants.csv`
- 更新表格中的数值
- 确保图片是最新的

## 图片文件清单

报告引用的图片文件（需要确保这些文件存在）：

### Performance Figures
- `outputs/figures/performance/CO_model_comparison.png`
- `outputs/figures/performance/best_models_across_horizons.png`
- `outputs/figures/performance/f1_score_all_models.png`
- `outputs/figures/performance/roc_auc_heatmap.png`
- `outputs/figures/performance/metrics_comparison_by_model_horizon.png`
- `outputs/figures/performance/model_ranking_table.png`

### Diagnostic Figures
- `outputs/figures/diagnostics/roc_pr_curves_top4.png`
- `outputs/figures/diagnostics/confusion_matrices_top4.png`
- `outputs/figures/diagnostics/feature_importance_CO.png`
- `outputs/figures/diagnostics/class_distribution_analysis.png`
- `outputs/figures/diagnostics/error_type_analysis.png`
- `outputs/figures/diagnostics/imbalance_ratio_heatmap.png`

## 自定义报告

### 添加更多章节

可以在相应位置添加新的 `\section{}` 或 `\subsection{}`。

### 修改样式

- 修改页面边距：调整 `\geometry{}` 参数
- 修改字体大小：修改 `\documentclass[12pt,...]` 中的 `12pt`
- 修改标题样式：使用 `\titleformat` 命令

### 添加表格

使用 `booktabs` 包创建专业表格：
```latex
\begin{table}[H]
\centering
\caption{表格标题}
\label{tab:label}
\begin{tabular}{@{}lcc@{}}
\toprule
列1 & 列2 & 列3 \\
\midrule
数据1 & 数据2 & 数据3 \\
\bottomrule
\end{tabular}
\end{table}
```

## 常见问题

### 1. 编译错误：找不到图片

**解决方法**：
- 检查图片文件路径是否正确
- 确保图片文件存在
- 使用绝对路径或相对路径

### 2. 参考文献不显示

**解决方法**：
- 运行 `bibtex report`
- 再运行两次 `pdflatex report.tex`

### 3. 中文显示问题

如果需要在报告中添加中文，在导言区添加：
```latex
\usepackage[UTF8]{ctex}
```

### 4. 表格太宽

**解决方法**：
- 使用 `\resizebox{\textwidth}{!}{...}` 缩放表格
- 或使用 `tabularx` 包自动调整列宽

## 建议

1. **使用版本控制**：将 `.tex` 文件加入 Git，但排除 `.pdf` 和临时文件
2. **定期备份**：编译前备份工作
3. **分章节编写**：可以将报告分成多个 `.tex` 文件，使用 `\input{}` 包含
4. **使用 Overleaf**：在线编辑和协作更方便

## 进一步定制

如果需要更专业的学术论文格式，可以考虑：
- 使用 `IEEEtran` 模板（IEEE 会议/期刊）
- 使用 `acmart` 模板（ACM 会议/期刊）
- 使用 `elsarticle` 模板（Elsevier 期刊）

这些模板可以从相应网站下载。







