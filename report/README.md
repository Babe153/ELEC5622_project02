# 报告写作交接入口

**给 Codex 的主入口：[CODEX_REPORT_TASK.md](CODEX_REPORT_TASK.md)。** 文件末尾有可直接复制到新对话的任务提示词。

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
