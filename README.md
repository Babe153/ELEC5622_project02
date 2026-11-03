# ELEC5622 Project 02：HEp-2 细胞分类

本项目已完成30轮探索、60轮训练与四组最终测试，以及100轮续训验证对照。**写报告请直接进入 [报告图表与数据入口](report/README.md)**：30、60、100轮的曲线、训练历史、指标表和报告大纲均已上传，无需下载数据集、模型权重或运行训练。完整检查点保留在本地 `results/` 和独立实验目录。
本机生成的 `results/check_report.json`（不提交 Git）是数据、权重加载和前向传播检查结果，不是作业实验结果，也不包含测试准确率。

## 目录结构

```text
project2/
├── README.md                 # 中文说明
├── config.json               # 实验设置
├── train.py                  # 检查、训练、测试
├── run.ps1                   # PowerShell 启动入口，默认只检查
├── requirements.txt          # 训练依赖
├── requirements-plot.txt     # 绘图依赖
├── tests/test_resume.py      # 合成数据的续训回归检查
├── .gitignore
├── datasets/                 # 本地数据，不上传 GitHub
│   ├── training/
│   ├── validation/
│   ├── test/
│   └── gt_training.csv
├── weights/
│   ├── alexnet-owt-7be5be79.pth
│   ├── resnet34-b627a593.pth
│   └── manifest.json
├── scripts/
│   ├── download_weights.py
│   └── plot_curves.py
├── references/
│   └── Hep2_cell_classification_IEEE_JBHI.pdf
└── results/                  # 自动生成，不上传 GitHub
    ├── check_report.json
    └── <实验名称>/           # 开始训练后才生成
```

GitHub 不包含原始细胞图片、数据集标签、模型权重二进制及论文 PDF；**包含 `report/` 中精选的实验结果、PNG/PDF图表和CSV/JSON数据**。
仅重新运行实验时，才需要将老师的数据放入 datasets，并下载权重。写报告不需要这些步骤。
`.git` 是本地版本历史，`__pycache__` 是 Python 自动缓存，图中省略。

## 现在已经具备的内容

- `weights/alexnet-owt-7be5be79.pth`：AlexNet IMAGENET1K_V1。
- `weights/resnet34-b627a593.pth`：ResNet34 IMAGENET1K_V1。
- `weights/manifest.json`：官方来源、大小、完整 SHA-256；下载时核对官方文件名中的哈希前缀。
- `train.py`：检查、训练调参、最终测试三个独立命令。
- `config.json`：实验方案，标签路径为 `datasets/gt_training.csv`。
- `scripts/download_weights.py`：下载脚本，始终把权重保存到项目根目录的 `weights/`。

官方说明：
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html

模型先加载完整的 ImageNet 1000 类权重，再换成随机初始化的六分类输出层。每次调参都重新开始，不继承上一次训练结果。正式训练从本地权重读取，无需联网下载。

## 实验方案

| 模型 | 微调策略 | 学习率 | 最大轮数 |
|---|---|---|---|
| AlexNet | 部分：classifier，即 FC6–FC8 | 0.01 / 0.001 / 0.0001 | 60 |
| AlexNet | 全量 | 同上 | 60 |
| ResNet34 | 部分：layer4 + fc | 同上 | 60 |
| ResNet34 | 全量 | 同上 | 60 |

共 12 次训练。SGD，momentum=0.9，weight_decay=0.0005，batch_size=16，固定学习率，无早停（非有限 loss 会终止对应试验），固定 seed=42。使用 CUDA 混合精度。只把 requires_grad=True 的参数传给优化器。

部分微调时，ResNet 冻结区域的 BatchNorm 运行统计量也保持不变。验证和测试均使用 eval 模式。AlexNet 保留 torchvision 模型原有 Dropout。

## 数据及预处理

使用 datasets 文件夹内老师提供的 training / validation / test 划分，不重新划分。代码验证数量、跨集合 ID 重复、CSV 标签完整性。类别名称排序生成 0–5 编码，保存到实验目录的 classes.json，避免人工猜测标签编号。

- 灰度图复制为 RGB 三通道。
- 图像直接缩放为 224×224，不做中心裁剪，以保留完整细胞视野。
- 训练时在线随机旋转 [-180°, 180°]，双线性插值，空白处填 0。
- 转为 [0,1]，按 ImageNet mean/std 标准化。
- 验证和测试不随机旋转。

训练采用在线随机旋转，每轮仍处理8,701个原始样本；输入直接缩放至224×224。报告应按实际代码描述预处理，不添加未使用的数据增强或模型集成。

## 如何运行

### PowerShell 启动脚本

在项目目录打开 PowerShell，使用以下命令。单独运行 `./run.ps1` 默认只检查，只有指定 `-Action train` 才启动训练。

```powershell
.\run.ps1 -Action check
# 准备好后再开始训练：
.\run.ps1 -Action train
# 所有验证集调参结束后再测试：
.\run.ps1 -Action test
```

可用 `-Experiment 实验名称`、`-Config 配置文件路径`、`-RunDir 结果目录` 指定实验参数。启动器默认使用项目内的 `.venv/Scripts/python.exe`，不读取其他环境路径、不自动回退到系统 Python。如果确需使用其他环境，可以通过 `-PythonExe` 显式指定。不会创建 environment 文件夹，也不会自动安装依赖。

训练窗口需保持打开。已有同名实验目录时程序拒绝覆盖；新实验支持从完整一轮的检查点恢复。测试不会在训练结束后自动执行。

### 直接使用 Python 命令

在 project2 文件夹打开终端，并选择已安装 requirements.txt 中依赖的 Python 环境。下面的 `python` 必须指向该环境；在 IDE 中也需要选择对应解释器。以下命令只是说明，准备代码时没有执行训练命令。

`train.py` 是核心执行入口，必须指定 `check`、`train`、`resume` 或 `test` 子命令。只执行 `python train.py` 会显示参数错误，不会开始训练。

检查（不会更新参数，也不预测测试集）：

```powershell
python train.py check
```

**确认要开始正式训练后**，运行 12 次实验：

```powershell
python train.py train --experiment baseline_60epochs
```

训练会打印每轮训练/验证准确率和耗时。已有同名实验目录时拒绝覆盖。若中断，使用下面的 resume 命令恢复；不要重复使用 train 命令覆盖已有实验。

如果想先运行某一种配置，复制 config.json 为 pilot.json，缩小 models、strategies 或 learning_rates 数组，然后：

```powershell
python train.py train --config pilot.json --experiment pilot
```

12 次训练全部结束、确认不再调参后，手动评估测试集：

```powershell
python train.py test --run-dir results/baseline_60epochs
```

测试命令只使用验证集选中的四个模型，不会测试其余学习率模型。成功生成 test_results.json 后拒绝重复测试，避免误把测试集用作调参工具。

## 如何选择最佳轮次和学习率

每轮评估验证集，先比较验证准确率；准确率相同则选择验证 loss 更低的。每种模型/微调策略在所有学习率及轮次中保留一个最佳模型。若二者都完全相同，保留先遇到的模型。

默认完整跑 60 轮。用各轮日志分析收敛情况，最佳权重可能早于最后一轮。每个学习率另存完整的最后一轮检查点，用于中断恢复或增加训练上限。

## 实验结果在哪里

`results/<实验名称>/`：

- `config.json`、`environment.json`、`weights.json`、`classes.json`：可复现配置。
- `trials.csv/json`：12 次调参的结果摘要。
- `selected_models.json`：四种配置的验证集赢家。
- 每个配置文件夹中，`lr_*_history.csv` 保存逐轮 loss、accuracy 和时间，可直接用于画曲线。
- `best.pt` / `selection.json`：该配置唯一保留的最佳权重及其来源。
- 最终测试后生成 `test_results.csv/json`、逐图预测及混淆矩阵 CSV（行是真实类别、列是预测类别）。

准确率保存为 0–1 比例，报告中乘 100 显示百分比。代码记录三种不同时间，报告中请明确口径：

1. `training_seconds_to_best`：选中学习率从第一轮到最佳轮的训练时间，不含验证和保存。
2. `selected_trial_training_seconds`：选中学习率实际跑完的所有轮次训练时间，不含验证和保存。
3. `group_tuning_wall_seconds`：该配置所有学习率试验循环的总墙钟时间，包含验证、日志和保存，不包含每次创建模型/加载预训练权重的准备时间。

每轮时间进行 CUDA 同步，包含读取数据、增强、传输及计算。第一次启动数据进程和 GPU 预热会影响首轮。代码为复现关闭 cuDNN benchmark，且有逐 batch 的非有限 loss 检查，因此时间可能比之前测速略长。固定随机种子提高复现性，但不承诺跨设备/软件版本逐位一致。

## 环境和空间

本机项目已配置独立的 `.venv`，`run.ps1` 默认调用其中的 Python。它不依赖其他项目或聊天工作目录。直接使用 `python` 命令时，仍需在终端或 IDE 选择该环境，也可以始终使用 `.\.venv\Scripts\python.exe`。

本机无需重新安装。GitHub 不包含 `.venv`；克隆项目到新电脑后，在项目根目录执行以下命令重建环境。虚拟环境不应直接复制到其他电脑：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不必激活环境，也可以直接指定 Python 来执行脚本：

```powershell
.\.venv\Scripts\python.exe scripts/download_weights.py
.\.venv\Scripts\python.exe train.py check
```

若使用已选择的环境，权重下载命令为 `python scripts/download_weights.py`。可用 `python -c "import sys; print(sys.executable)"` 查看当前 Python 位置，用 `python -c "import torch; print(torch.cuda.is_available())"` 确认 CUDA 可用。GPU 模式是默认设置；不会静默切换为 CPU。

脚本与配置中的数据、权重、结果路径以项目根目录为基准，不依赖当前终端所在目录。

每种配置保存一个最佳模型，同时每个学习率保存一个可覆盖更新的最后轮检查点（包含 SGD 动量）。训练前检查剩余磁盘是否能容纳所有检查点、原子替换临时文件及安全余量；空间不足会在训练开始前报错。运行时请插电，并避免同时运行占用 GPU 的任务。

提交代码时按老师要求排除图片数据；通常也不需要打包已下载的预训练权重或 Python 环境，保留下载脚本、配置、依赖和实际实验代码即可。报告中的结果必须来自之后真正完成的实验。

## 绘制训练曲线

绘图脚本只读取已有 CSV/JSON，不启动训练，也不评估测试集。首次使用安装独立的绘图依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plot.txt
.\.venv\Scripts\python.exe scripts/plot_curves.py --run-dir results/baseline_60epochs
```

输出位于实验目录的 `plots/`：所有学习率的验证准确率总览、选中学习率的训练/验证准确率及 loss，以及每个配置的全部学习率详细图。每张图均提供 PNG 和 PDF；星号/竖虚线标示最佳验证轮次。曲线使用原始数据，不做平滑。发散且没有完整轮次的实验明确标注，不伪造曲线。


## 完整续训（仅适用于本版本新建的实验）

新实验默认 `baseline_60epochs`，从 ImageNet 权重开始；旧 `baseline_30epochs` 不会被修改，也不能补造其未保存的优化器/随机状态。

```powershell
# 中断后恢复：读取实验保存的配置，而不是当前 config.json
.\run.ps1 -Action resume -Experiment baseline_60epochs
# 已完成 60 轮后，增加到总计 100 轮（不是额外 100 轮）
.\run.ps1 -Action resume -Experiment baseline_60epochs -Epochs 100
# 等价的 Python 入口
.\.venv\Scripts\python.exe train.py resume --run-dir results/baseline_60epochs --epochs 100
```

每个组合的 `lr_*_last.pt` 保存模型参数与 BatchNorm 缓冲区、SGD 状态（含 momentum）、GradScaler、Python/NumPy/PyTorch/CUDA 随机状态、独立的 shuffle sampler generator 与两个 DataLoader 的 worker generator 状态、已完成轮次、历史和累计耗时。`best.pt` 继续用于验证集选模。没有学习率调度器，因此没有 scheduler 状态需要保存。

只在整轮训练及验证完成后保存，使用临时文件加原子替换。中途退出时恢复到上一完整轮，重做未完成轮次。已达到目标轮数的试验会跳过，已标记发散的试验不会自动重试。完整续训禁止更改模型、学习率和数据等设置，只允许增加总轮数；运行环境变化或已评估测试集的实验会拒绝续训。请保持数据内容不变。

新版复用 persistent workers，避免 Windows 每轮重新启动数据加载进程。训练 sampler 将轮次与图片索引一起送入 worker；旋转角度由固定种子、轮次与图片编号确定，仍在 [-180°, 180°] 均匀采样，不依赖 worker 的随机状态或预取顺序。shuffle sampler 使用独立 generator，其状态保存在检查点中，避免恢复后 worker 重建影响抽样顺序。

v1（每轮重启 worker）的检查点首次升级需显式指定 `-UpgradeLoader` 或 `--upgrade-loader`。升级保留已有权重、优化器动量、GradScaler、全局随机状态与历史；旧 loader generator 状态作为新 sampler 的起始状态。后续随机旋转与抽样序列会改变，记录在 `loader_upgrade.json` 和各 trial 的升级记录中。之后使用普通 resume 即可。旧 30 轮实验缺少完整状态，仍不能用于此升级。跨硬件或非确定性 CUDA 算子仍不保证逐位复现。

当前 60 轮实验在第 7 轮后升级；前 7 轮的权重及完整状态已保留，备份在该实验的 `before_loader_upgrade/`。记录原始耗时，不把优化前后的计时视为完全一致的加载条件。

延长训练会更新同一个新实验目录的配置和汇总，历史 CSV 保留此前轮次；若需要独立留档，应先备份该新实验目录。训练不自动进行测试。绘图横轴按实际记录轮次自适应，尾部统计使用最后五轮与此前五轮，短记录的空窗口输出 null。


续训回归检查：`.\.venv\Scripts\python.exe tests/test_resume.py`。仅用临时合成图片和小型 CPU 网络，验证连续运行、中断恢复和增加轮数的结果一致，以及旧实验/已测试实验/磁盘不足的保护逻辑；不读取作业数据，不训练 AlexNet 或 ResNet34。


### 安全暂停

在实验目录创建空文件 `pause.request`，程序会在下一轮开始前（上一轮已保存）退出，生成 `paused.json`，不会标记训练完成。删除 `pause.request` 后再执行 resume。若直接关闭窗口，仍能回退到最后一个完整检查点，但可能重做未完成的轮次。CSV 与 PT 都使用临时文件加原子替换，恢复时以 PT 内历史为准。

## 报告写作交接

60轮实验及最终测试已完成。队友可直接从 [report/README.md](report/README.md) 开始阅读；[详细大纲](report/REPORT_OUTLINE_ZH.md) 包含作业要求、逐节写法、图表和实际数据；[Codex执行说明](report/CODEX_REPORT_TASK.md) 可直接交给写作助手。`report/` 是可提交Git的轻量证据副本，原始 `results/` 仍保持本地、不上传模型检查点。

## 额外100轮续训准备

独立续训副本、空间计算、暂停状态和将来的启动命令见 [docs/100epoch_extension.md](docs/100epoch_extension.md)。准备工具 `scripts/prepare_extension.py` 默认只检查空间；只有 `--prepare` 才复制完整状态，任何模式都不会训练。


## 100轮补充实验结果

独立续训至100轮已完成，未再次运行测试集。保留原60轮最终测试结果，新增 [60→100轮验证对照](report/tables/60_vs_100_comparison.md) 和 [100轮实验记录/曲线](report/experiments/baseline_100epochs)。大纲与Codex执行说明已同步加入这一补充分析。

仅导出已保存结果（不训练、不测试）：`python scripts/export_extension_results.py --run-dir <100轮实验目录>`。
