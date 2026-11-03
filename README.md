# ELEC5622 Project 02：HEp-2 细胞分类

本项目已准备好训练代码；本机已下载官方 ImageNet 预训练权重，GitHub 克隆后请运行下载脚本获取。**尚未开始正式训练。**
本机生成的 `results/check_report.json`（不提交 Git）是数据、权重加载和前向传播检查结果，不是作业实验结果，也不包含测试准确率。

## 目录结构

```text
project2/
├── README.md                 # 中文说明
├── config.json               # 实验设置
├── train.py                  # 检查、训练、测试
├── requirements.txt          # Python 依赖
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
│   └── download_weights.py
├── references/
│   └── Hep2_cell_classification_IEEE_JBHI.pdf
└── results/                  # 自动生成，不上传 GitHub
    ├── check_report.json
    └── <实验名称>/           # 开始训练后才生成
```

GitHub 不包含图片、标签 CSV、权重二进制、论文 PDF和运行结果。
克隆后将老师的数据按上述结构放入 datasets，再下载权重即可。
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
| AlexNet | 部分：classifier，即 FC6–FC8 | 0.01 / 0.001 / 0.0001 | 30 |
| AlexNet | 全量 | 同上 | 30 |
| ResNet34 | 部分：layer4 + fc | 同上 | 30 |
| ResNet34 | 全量 | 同上 | 30 |

共 12 次训练。SGD，momentum=0.9，weight_decay=0.0005，batch_size=16，固定学习率，无早停（非有限 loss 会终止对应试验），固定 seed=42。使用 CUDA 混合精度。只把 requires_grad=True 的参数传给优化器。

部分微调时，ResNet 冻结区域的 BatchNorm 运行统计量也保持不变。验证和测试均使用 eval 模式。AlexNet 保留 torchvision 模型原有 Dropout。

## 数据及预处理

使用 datasets 文件夹内老师提供的 training / validation / test 划分，不重新划分。代码验证数量、跨集合 ID 重复、CSV 标签完整性。类别名称排序生成 0–5 编码，保存到实验目录的 classes.json，避免人工猜测标签编号。

- 灰度图复制为 RGB 三通道。
- 图像直接缩放为 224×224，不做中心裁剪，以保留完整细胞视野。
- 训练时在线随机旋转 [-180°, 180°]，双线性插值，空白处填 0。
- 转为 [0,1]，按 ImageNet mean/std 标准化。
- 验证和测试不随机旋转。

这是参考论文旋转增强思想的在线方案，不是原文固定 9°/18°/36°枚举的严格复现，每轮仍然只有 8,701 个训练样本。这里也没有采用论文的 78×78 网络、逐图 min-max 归一化、PCA 对齐、测试时增强或多个轮次模型集成。预训练权重官方推理示例采用 resize 256 + center crop 224；本项目为细胞数据统一采用直接 resize 224，需在报告中明确这一选择。旋转会裁去越界区域，正式训练前可进一步检查增强样例是否保留了细胞结构。

## 如何运行

在 project2 文件夹打开终端，并选择已安装 requirements.txt 中依赖的 Python 环境。下面的 `python` 必须指向该环境；在 IDE 中也需要选择对应解释器。以下命令只是说明，准备代码时没有执行训练命令。

`train.py` 是唯一运行入口，必须指定 `check`、`train` 或 `test` 子命令。只执行 `python train.py` 会显示参数错误，不会开始训练。

检查（不会更新参数，也不预测测试集）：

```powershell
python train.py check
```

**确认要开始正式训练后**，运行 12 次实验：

```powershell
python train.py train --experiment baseline_30epochs
```

训练会打印每轮训练/验证准确率和耗时。已有同名实验目录时拒绝覆盖。若中断，已有日志和最佳权重保留，但本版本不支持精确断点续训；重新训练需要使用新的实验名称。

如果想先运行某一种配置，复制 config.json 为 pilot.json，缩小 models、strategies 或 learning_rates 数组，然后：

```powershell
python train.py train --config pilot.json --experiment pilot
```

12 次训练全部结束、确认不再调参后，手动评估测试集：

```powershell
python train.py test --run-dir results/baseline_30epochs
```

测试命令只使用验证集选中的四个模型，不会测试其余学习率模型。成功生成 test_results.json 后拒绝重复测试，避免误把测试集用作调参工具。

## 如何选择最佳轮次和学习率

每轮评估验证集，先比较验证准确率；准确率相同则选择验证 loss 更低的。每种模型/微调策略在所有学习率及轮次中保留一个最佳模型。若二者都完全相同，保留先遇到的模型。

默认完整跑 30 轮，通过 history.csv 判断是否需要更多轮数，无需另起 20 轮实验。若要增加到 50 轮，可更改 epochs、使用新的实验名称重新运行对应候选配置。本版本不保存每个试验最后一轮的优化器状态，因此不能无缝接着第 30 轮训练；这是节省本机磁盘空间的取舍。

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

项目不保存 Python 解释器的本机路径。Python 环境由终端或 IDE 选择。删除启动脚本和路径配置不会卸载此前安装的环境，也不会自动让系统默认的 `python` 具备 PyTorch。

如果已有兼容的 GPU Python 环境，选择它后直接执行上面的命令即可。若需要新建项目环境，可以在项目根目录执行（本次整理没有执行安装）：

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

每种配置只保存一个最佳模型，四个 FP32 模型合计约 0.63 GB，原始预训练权重合计约 0.33 GB。保存时还需临时空间，不会保存数十个 AlexNet 大文件。运行时请插电，并避免同时运行占用 GPU 的任务。

提交代码时按老师要求排除图片数据；通常也不需要打包已下载的预训练权重或 Python 环境，保留下载脚本、配置、依赖和实际实验代码即可。报告中的结果必须来自之后真正完成的实验。
