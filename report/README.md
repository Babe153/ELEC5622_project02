# 报告写作交接入口

**给 Codex 的主入口：[CODEX_REPORT_TASK.md](CODEX_REPORT_TASK.md)。** 文件末尾有可直接复制到新对话的任务提示词。

## 按轮次直接取图和数据

本目录就是原始 `results/` 的报告用精简导出，图表和数据已提交到GitHub；不含模型检查点。PNG可直接插入Word，PDF可用于矢量排版，CSV可用于核对数值或重新绘图。

| 实验 | 推荐总览图 | 各模型、各学习率的完整曲线 | 原始指标与选择记录 |
|---|---|---|---|
| 30轮早期探索 | [训练/验证曲线](experiments/baseline_30epochs/plots/selected_training_curves.png) · [验证比较](experiments/baseline_30epochs/plots/validation_comparison.png) | [PNG/PDF目录](experiments/baseline_30epochs/plots/) | [实验数据目录](experiments/baseline_30epochs/) · [候选结果CSV](experiments/baseline_30epochs/trials.csv) |
| 60轮正式实验 | [训练/验证曲线](experiments/baseline_60epochs/plots/selected_training_curves.png) · [验证比较](experiments/baseline_60epochs/plots/validation_comparison.png) | [PNG/PDF目录](experiments/baseline_60epochs/plots/) | [实验数据目录](experiments/baseline_60epochs/) · [候选结果CSV](experiments/baseline_60epochs/trials.csv) |
| 100轮续训对照 | [训练/验证曲线](experiments/baseline_100epochs/plots/selected_training_curves.png) · [验证比较](experiments/baseline_100epochs/plots/validation_comparison.png) | [PNG/PDF目录](experiments/baseline_100epochs/plots/) | [实验数据目录](experiments/baseline_100epochs/) · [候选结果CSV](experiments/baseline_100epochs/trials.csv) |

每个实验下的四个模型子目录均包含逐轮 `*_history.csv`（训练/验证loss、accuracy和时间）及 `selection.json`；实验根目录的 `selected_models.json` 记录验证选择结果。

## 正文优先使用的汇总材料

- [60轮最终结果表](tables/final_comparison.md) / [CSV](tables/final_comparison.csv)：测试准确率、可训练参数数量和训练时间。
- [测试混淆矩阵PNG](figures/test_confusion_matrices.png) / [PDF](figures/test_confusion_matrices.pdf)。逐图预测与混淆矩阵CSV在60轮对应模型目录。
- [准确率、参数和时间比较PNG](figures/accuracy_parameters_time.png) / [PDF](figures/accuracy_parameters_time.pdf)。
- [60与100轮对比图PNG](figures/extension_60_vs_100.png) / [PDF](figures/extension_60_vs_100.pdf)，配套[解读与表格](tables/60_vs_100_comparison.md)、[四组选择结果CSV](tables/60_vs_100_selected.csv)、[所有候选CSV](tables/60_vs_100_all_trials.csv)。
- [类别分布CSV](tables/class_distribution.csv)、[每类测试指标CSV](tables/per_class_test_metrics.csv)。
- [同一次60轮运行内的前30轮与前60轮对比CSV](tables/within_run_30_vs_60.csv)：这是同一训练轨迹的比较，不是独立30轮实验与60轮实验的直接续训关系。

**结果口径：只有60轮实验有最终测试结果。30轮用于早期探索，100轮只用于验证集上的补充对照；不能把60轮测试成绩写成100轮模型的测试成绩。此交接材料不包含150轮实验结果。**

下载方式：在仓库首页选择 **Code → Download ZIP**，解压后直接使用 `report/`。已有克隆的队友可执行 `git pull` 更新，不需要安装Python环境。

## 写作说明

- [逐节详细大纲](REPORT_OUTLINE_ZH.md)：写什么、数据口径、图表安排、讨论和提交要求。
- [最终结果表](tables/final_comparison.md)：四组最终测试、参数及训练时间。
- [60→100轮验证对照](tables/60_vs_100_comparison.md)：补充实验的增益、轮次和成本；100轮没有测试结果。
- `experiments/baseline_100epochs/plots/`：100轮完整曲线；`figures/extension_60_vs_100.pdf` 为预算对比图。
- `experiments/baseline_60epochs/plots/`：最终训练曲线，PNG/PDF。
- `figures/`：测试混淆矩阵、准确率/参数/时间比较。
- `tables/`：类别分布、每类指标、同一运行的30/60轮比较。
- `experiments/`：已完成实验的配置、历史、选择记录、测试预测及原始结果。

仓库公开可读，报告撰写无需本机数据、模型权重或Python环境，也无需重新训练/测试。保留60轮实验的四个验证最佳模型对应的最终测试结果；100轮是独立续训的验证补充对照，不替换原测试主表；30轮记录只作早期探索。

执行顺序：读取证据 → 核验数字 → 按五节大纲写作 → 插图引用 → 排版自检 → 交付。具体路径、数据检查规则和输出要求均在执行说明中。

正常保留背景和理论文献引用；不加入课堂/课件页码映射、课堂案例成绩或与论文结果对照的内容。

每位成员需按老师要求独立提交，并填写真实贡献。

重新导出已有结果（仅本机有原始results时使用）：

```powershell
.\.venv\Scripts\python.exe scripts/export_report_assets.py
```

`file_manifest.json` 为交接文件的SHA-256校验清单。
