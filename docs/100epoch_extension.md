# 独立100轮续训实验（已完成，未再次测试）

- 原实验：项目内 `results/baseline_60epochs/`，保留完整模型、历史及最终测试结果。
- 新实验：`D:/ELEC5622_project02_results/baseline_100epochs/`。
- 代码、原始数据及预训练权重仍在原项目；新实验仅将结果/检查点放D盘。
- 新目标是总计100轮；11个正常完成的候选从第61轮继续，原来发散的1个候选保持终止。
- 使用每个候选的 `*_last.pt`，不是各组的 `best.pt` 来恢复。模型、SGD动量、GradScaler、全局随机状态及独立采样器状态均保留；原best.pt也复制到新目录作为前60轮验证最佳基线。
- 采用独立物理副本，不使用硬链接/符号链接，后续更新不会覆盖60轮结果。
- 新目录没有顶层完成标记和测试结果；原测试结果仅在 `baseline_60_summary/` 中作为来源证据保留。`extension_provenance.json` 明确记录原实验已经使用过测试集；新实验属于额外探索性验证比较。
- `prepared.json` 是准备时的历史记录；实际运行已完成，当前以 `completed.json` 为准。未执行100轮测试。

## 空间

准备时核算：副本约3.967 GiB，原子保存临时文件约0.425 GiB，安全/历史增长余量0.531 GiB，总计约4.924 GiB可用空间。C盘当时约1.720 GiB，不足约3.203 GiB；因此新副本放D盘，不删除或移动C盘文件。

重新核算一个尚不存在的目标目录（只读，不复制，不训练）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_extension.py --destination D:/ELEC5622_project02_results/another_extension --epochs 100
```

添加 `--prepare` 才会复制并校验文件，仍不会开始训练。已有目标目录会拒绝覆盖。准备会逐个检查完整检查点结构、已保存轮次与状态，并校验源文件/副本SHA-256。没有复制中途升级前备份、图片、环境或原实验图表。

## 当时的启动方式（历史记录，不需再次执行）

准备阶段未启动训练，之后按用户指令删除暂停标记并恢复，现已完成。以下保留为流程记录：

```powershell
Remove-Item -LiteralPath 'D:/ELEC5622_project02_results/baseline_100epochs/pause.request'
.\run.ps1 -Action resume -RunDir 'D:/ELEC5622_project02_results/baseline_100epochs'
```

不要使用 `-Action train`，也不要把默认配置改成100轮后对旧实验重跑。`resume` 会读取新目录内epochs=100的配置。预计补充运行约3.5–4小时。

完成后主要比较同一候选前60轮与前100轮最佳验证结果和新增训练时间。因为原实验已评估测试集，不把这次延长包装成未接触测试结果的确认性实验，也不自动再次评估测试集。

新实验保留了以前的累计时间字段；额外成本应以第61–100轮的history逐行求和，不能把累计100轮时间误称为新增40轮时间。

100轮结果及60→100轮对比已导出到 `report/experiments/baseline_100epochs/` 和 `report/tables/60_vs_100_comparison.md`。记录的新增循环wall time约3.16小时。
