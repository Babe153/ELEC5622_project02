# 给报告撰写同学的入口

1. 先读 [详细大纲](REPORT_OUTLINE_ZH.md)：逐节写什么、使用哪张图、数据来源、课件页码、限制和提交要求。
2. 核心数据看 [最终比较表](tables/final_comparison.md)，可直接用于报告；其他CSV均为UTF-8。
3. 最终实验：`experiments/baseline_60epochs/`；旧30轮仅作探索记录，未做测试。
4. `figures/`：最终测试混淆矩阵及准确率/参数/时间比较，PNG供预览、PDF供报告排版。
5. `experiments/baseline_60epochs/plots/`：学习率总览、四个选中候选的train/val曲线、每个配置全部候选详细图。
6. `tables/`：类别分布、每类测试指标、同一次60轮实验内前30轮与全60轮的验证比较。
7. 每个模型目录含history、选择记录、测试预测、混淆矩阵；无需下载图片或运行模型即可写结果分析。

结果来源于真实运行，逐图预测与混淆矩阵已交叉核验。正式结论使用60轮实验按验证集选出的四个模型；测试均已完成，不需要重新训练或重新测试。

本目录不含原始图片、标签源CSV、Python环境、预训练权重或大型模型检查点。数据在本机，若需要重现实验需按根README单独准备。课件及作业PDF未复制到仓库，请使用课程提供的版本。

如需重新导出报告材料（只读取结果，不训练、不评估测试集）：

```powershell
.\.venv\Scripts\python.exe scripts/export_report_assets.py
```

图表脚本为 `scripts/plot_curves.py`；导出脚本为 `scripts/export_report_assets.py`。`file_manifest.json` 记录交接材料文件大小与SHA-256。

这份材料是共享实验依据。老师要求每人独立提交报告并说明成员贡献，不能用同一份报告替代个人提交。
