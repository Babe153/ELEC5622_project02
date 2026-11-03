# 报告详细大纲与写作交接（以最终 60 轮实验为准）

这是一份写作依据，不是已经完成的个人报告。每位成员应理解实验、独立组织文字，并填写真实贡献。所有结果均来自本仓库保存的实验输出；不要编造未做过的实验。先阅读本文，再打开 `tables/final_comparison.md` 和图表。

## 0. 必须遵守的作业要求

- 任务：HEp-2 六分类；ImageNet 预训练 AlexNet 与 ResNet34；每种模型比较局部/全量微调，形成四种配置。
- 沿用提供的训练/验证/测试划分；验证集选超参数和最佳轮次；测试集用于最终评估。
- 必须比较 **accuracy、trainable parameter counts、training time**，讨论架构和微调策略。
- 每人提交个人报告，不是小组只交一份。在报告开头清楚列出成员贡献；正文也保留老师建议的 Section 2。
- 提交 PDF 或 DOC/DOCX 报告，以及 ZIP/RAR/7Z 代码包；提供依赖说明，不包含数据集。大型模型与环境通常不必打包。
- 教师建议五部分：1 Aims and Background；2 Contributions of Team Members；3 Methodology；4 Results；5 Discussion and Conclusion。
- 原始作业 PDF 明确要求在 Section 5 的末尾出现一次 `KAANet`。这是文档中的格式要求，不是我们训练过的模型；不要把它写进模型比较或虚构对应实验。
- PDF 截止时间写为 2026-11-08 23:59，同时标注 Week 13 Friday，日期与星期不一致，提交前以 LMS/老师通知核实。
- 评分：Introduction/Aims 5%；Background and Methods 25%；Results 50%；Discussion and Conclusion 20%。优先把结果组织、数据来源和讨论写扎实。
- 原文未给出明确页数/字数上限。可暂按 8–12 页正文（含图表）规划；这只是写作建议，不是老师规定。

## 1. Aims and Background（建议约 10% 正文）

### 1.1 任务背景

写 HEp-2 细胞染色模式分类为何需要自动化：人工判读存在主观性、工作量和一致性问题；CNN 从图像学习判别特征。强调本项目是提供数据上的图像分类实验，不宣称已经能独立诊断疾病或部署临床。

交代六类：Centromere、Golgi、Homogeneous、NuMem、Nucleolar、Speckled。NuMem 的显示名称沿用标签文件，不擅自重命名编码。

参考教师提供的 Gao 等人的 HEp-2 论文及 Week11 第21页。可以放少量来自训练集的示例图，但本交接包没有上传原始图片；如自行加入须标清来源，不从测试集挑图调模型。

### 1.2 理论背景

- CNN：卷积提取局部模式，逐层形成更复杂特征，分类层输出六类 logits。
- 预训练迁移：从 ImageNet 权重初始化，替换原1000类输出层，不是从随机权重训练整个网络。
- AlexNet：卷积层后接较大的全连接分类器；本项目使用 torchvision 实现，保留其 Dropout。
- ResNet34：残差块采用 H(x)=F(x)+x，使较深网络更容易优化；不能把“更深必然更好”写成定律。
- 局部与全量微调：冻结部分层降低可训练参数及更新成本；全量允许各层适应目标域，但也可能过拟合或不稳定。

### 1.3 明确研究问题

建议以三点结束本节：
1. 同一划分与优化设置下，AlexNet 和 ResNet34 的分类效果如何？
2. 局部/全量微调在准确率、可训练参数和训练耗时之间如何权衡？
3. 学习率和训练轮数如何影响收敛、稳定性及验证集选模？

不要提前用测试结果反向定义研究目标。可以表述合理假设，但不能声称实验已证明某个架构机制的因果作用。

## 2. Contributions of Team Members（开头也需列简表）

用真实姓名/学号和具体工作填写，不使用虚构比例或默认均分。

| 姓名/学号 | 实际承担工作 | 可核验产物 |
|---|---|---|
| 待填写 | 例如：数据检查、训练代码实现、实验执行 | 对应脚本/实验记录 |
| 待填写 | 例如：文献与课件梳理、结果分析、报告撰写 | 对应章节/图表 |

每个人独立提交时说明共同完成部分与个人贡献。若课程另有工具使用披露要求，按实际使用情况填写。

## 3. Methodology（建议约 25–30% 正文）

### 3.1 数据集与划分

总计 13,596 张：training 8,701，validation 2,175，test 2,720。没有重新随机划分。用 CSV 的 Image ID 对应文件名，并检查标签齐全和三个划分 ID 不重叠。

**表1：数据分布**，直接取 `tables/class_distribution.csv`：

| 类别 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| Centromere | 1791 | 463 | 487 |
| Golgi | 449 | 110 | 165 |
| Homogeneous | 1556 | 415 | 523 |
| NuMem | 1430 | 337 | 441 |
| Nucleolar | 1675 | 394 | 529 |
| Speckled | 1800 | 456 | 575 |

指出类别并非完全均衡，故除了总体准确率，可补充每类召回率、macro-F1 和混淆矩阵。未采用类别加权或重采样，不要写成使用过。

### 3.2 预处理与数据增强

按实际代码顺序描述：
1. 转成 RGB 三通道（灰度图复制通道），直接双线性缩放为 224×224。
2. 仅训练集采用在线随机旋转，角度在 [-180°,180°] 均匀取值、双线性插值、边界填0。
3. ToTensor 将数值缩放到 [0,1]，用 ImageNet mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225] 标准化。
4. 验证/测试无随机旋转，使用相同 resize 与 normalization。

每轮仍然处理8,701张训练图片，不是把数据离线扩增为10/20/40倍。没有水平翻转、随机裁剪、PCA、测试时增强或模型集成。

**与参考论文的关系**：借鉴旋转增强思想，但没有严格复现论文固定角度步长生成旋转样本、不同网络和其他组件。不能直接把论文结果视为本实验复现基准。

### 3.3 模型与四种微调配置

**表2：配置与可训练参数**：

| 模型/策略 | 更新哪些层 | 可训练参数 |
|---|---|---:|
| AlexNet partial | classifier，即FC6–FC8 | 54,558,726 |
| AlexNet full | 全部层 | 57,028,422 |
| ResNet34 partial | layer4 + fc | 13,117,446 |
| ResNet34 full | 全部层 | 21,287,750 |

参数计数为 requires_grad=True 的标量元素总数；不是层数，不是模型文件大小。

说明：先加载官方 ImageNet IMAGENET1K_V1 的1000类权重，再将末层替换成随机初始化的6类线性层。局部微调的冻结层不更新参数；冻结的 ResNet BatchNorm 模块也保持 eval，避免更新运行统计量。所有验证/测试均 model.eval()，AlexNet Dropout 此时关闭。torchvision 标准 ResNet34 没有 AlexNet 式 Dropout。

课件的 AlexNet 示意图是经典版本；我们实际使用 torchvision 版本及224输入，不声称与课件227输入、96首层通道/LRN的图逐层完全一致。

### 3.4 训练与超参数选择

**表3：统一训练设置**：

| 设置 | 实际值 |
|---|---|
| 损失 | 六分类交叉熵，对原始logits计算 |
| 优化器 | SGD |
| 候选学习率 | 0.01、0.001、0.0001 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Batch size | 16 |
| 正式实验预算 | 每个候选最多60轮 |
| 随机种子 | 42 |
| 数值计算 | CUDA autocast + GradScaler 混合精度 |
| 调度器 | 无，固定学习率 |
| 常规早停 | 无 patience-based early stopping |
| 异常处理 | 非有限loss结束该候选，继续下一候选 |

每个配置尝试3个学习率，共12次候选训练。每个候选从同一ImageNet预训练起点重新初始化，只有同一次候选中断恢复时才继承状态。每轮：训练集前向/反向更新 → 完整验证集评估 → 保存记录与检查点。

验证选择规则：准确率更高优先；相同时选择验证loss更低；两者完全相同时保留先遇到的模型。跨该组所有学习率和轮次保存一个 best.pt，四组共四个最终选中模型。last.pt 用于完整续训，两者用途不同。最后一轮不一定是最佳轮。

### 3.5 30轮探索、60轮正式实验及实现优化

先做30轮探索；因若干候选后期仍在改善，决定扩展训练预算。从预训练起点重新做60轮，不是把旧30轮 best.pt 接着训练30轮。旧实验未保存完整续训状态。

60轮实验最初关闭persistent workers以保存可恢复随机流程；AlexNet partial、lr=0.01 在第7轮后保存并暂停，优化为复用worker，并按种子/轮次/图片ID确定旋转、独立保存采样RNG，从第8轮接续。保留权重、SGD动量、GradScaler和前7轮记录；后续随机序列有变化。其余候选从起点使用新版数据加载流程。保存升级记录供复现，不要声称旧30轮与新60轮前30轮逐点相同。

正文用2–3句说明即可，详细实现放附录。比较延长轮数的效果时，优先使用 **同一60轮运行的前30轮最佳值 vs 全60轮最佳值**（`tables/within_run_30_vs_60.csv`），而不是把两次不同随机流程的实验当成严格对照。

### 3.6 评估指标、计时与环境

- Accuracy = 正确预测数 / 总图片数；主要报告测试accuracy，验证accuracy单独列。
- 混淆矩阵：行真实类别、列预测类别。图中色彩按行归一化、格内数字是计数，不能混淆。
- Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1为二者调和平均；macro-F1为六类F1的等权平均。来自同一批保存的预测，不是重新推理。
- 主表训练时间建议用所选学习率完整60轮的 training_seconds/60：不含验证、保存与其他学习率搜索时间；包括训练批次数据加载、传输和计算。
- 辅助列可给达到最佳轮次的累计训练时间；另表给全部学习率搜索的 trial wall time。不要混用不同口径。
- group_tuning_wall_seconds 包含训练、验证和部分保存/日志开销，不包含每个候选开始前模型构建、加载权重，也不包括人工暂停时长。中断边界计时可能有小幅记账误差，不把它当作精确端到端挂钟时间。
- 第一轮有GPU/worker启动开销；初始7轮的数据加载流程不同。时间比较是本机实现条件下的观察，不是架构固有速度定律。
- GPU：RTX3050 Laptop，4GB；Python3.10.0、torch2.14.1+cu126、torchvision0.29.1+cu126、CUDA12.6（以保存的environment.json为准）。CPU Ryzen5600H，内存32GB。固定seed和设置不保证跨机器逐位复现。

## 4. Results（建议约 35–40% 正文，评分占50%）

顺序建议“超参数选择 → 曲线与收敛 → 最终测试 → 参数和时间 → 错误分析”。先写事实，解释主要留到Section5。

### 4.1 12次候选训练与验证选模

使用 `experiments/baseline_60epochs/trials.csv`：列配置、学习率、完成轮数、状态、最佳验证轮次、最佳验证accuracy。不能只报告四个成功最佳值、隐去不稳定的候选。

本次11次跑满60轮；AlexNet full、lr=0.01 完成2轮后因非有限loss停止。它是数值发散，不是根据验证集触发的早停，也不是“只设置了2轮”。AlexNet partial、lr=0.01 跑满60轮，但后期性能退化。

**图1：三种学习率验证曲线**：`experiments/baseline_60epochs/plots/validation_comparison.pdf`。星号是各候选最佳验证轮次。指出0.001总体较好、0.0001较慢、高学习率对AlexNet不稳定。

### 4.2 收敛与轮数选择

**图2：四个选中候选的train/val accuracy与loss**：`experiments/baseline_60epochs/plots/selected_training_curves.pdf`。蓝=train，橙=val，虚线=最佳验证轮次。未平滑。

分析：训练/验证总体先改善后趋缓；全量微调与ResNet partial后段训练loss继续下降，验证loss改善有限，存在泛化差距。不能仅因train accuracy低于val就断言数据错误：训练有旋转、Dropout且边更新边统计，验证使用epoch末模型、无增强及eval模式。

可选 **表4：同一次运行30轮与60轮的预算比较**，选0.001这一行展示：

| 配置 | 前30轮最佳验证 | 全60轮最佳验证 | 增量（百分点） |
|---|---:|---:|---:|
| AlexNet partial | 95.22% | 95.91% | +0.69 |
| AlexNet full | 97.10% | 97.79% | +0.69 |
| ResNet34 partial | 96.64% | 97.52% | +0.87 |
| ResNet34 full | 98.57% | 99.08% | +0.51 |

这是描述性比较，不是多个随机种子均值，也不能把“从更多轮次中取最大值”的上升当成严格统计显著的提升。

### 4.3 最终测试结果——核心表

四组都选择lr=0.001，分别加载各自验证最佳权重，仅对同一份2720张测试集做最终推理。

| 配置 | 最佳轮次 | 验证accuracy | 测试accuracy | 正确/总数 | 错误 |
|---|---:|---:|---:|---:|---:|
| AlexNet partial | 43 | 95.91% | 94.04% | 2558/2720 | 162 |
| AlexNet full | 44 | 97.79% | 96.62% | 2628/2720 | 92 |
| ResNet34 partial | 51 | 97.52% | 96.47% | 2624/2720 | 96 |
| ResNet34 full | 50 | 99.08% | 98.31% | 2674/2720 | 46 |

来源 `tables/final_comparison.csv`。99.08%是验证准确率，不能写成测试准确率。所有预测行数、唯一ID、混淆矩阵计数与总体准确率已交叉核对。

### 4.4 参数量与训练时间——作业明确要求

| 配置 | 可训练参数（百万） | 到最佳轮训练（分钟） | 所选候选完整60轮训练（分钟） | 全部学习率调参wall time（分钟） |
|---|---:|---:|---:|---:|
| AlexNet partial | 54.56 | 11.87 | 16.57 | 61.24 |
| AlexNet full | 57.03 | 15.33 | 20.77 | 48.57 |
| ResNet34 partial | 13.12 | 15.76 | 18.51 | 67.07 |
| ResNet34 full | 21.29 | 33.72 | 40.40 | 130.26 |

**图3** `figures/accuracy_parameters_time.pdf`：三个面板比较测试准确率、参数和所选候选60轮训练时间。

特别解释：AlexNet full总调参时间比partial短，是因为lr=0.01候选提前发散，不能据此断言全量微调单轮更快。AlexNet partial大部分参数仍在FC分类器，冻结卷积层只减少约4.33%可训练参数；ResNet partial约减少38.38%。参数量与计算量/运行时间并非线性等价，因此ResNet34 full参数更少但训练更久并不矛盾。

### 4.5 混淆矩阵与类别差异

**图4** `figures/test_confusion_matrices.pdf`；每类指标来自 `tables/per_class_test_metrics.csv`。解释一两组最常见错分方向，务必查看实际格子，不凭主观想象。

实际错分例子：AlexNet partial 的 Speckled→Homogeneous 为41张；AlexNet full为24张；ResNet partial为18张；ResNet full为8张。同一错分方向逐步减少，可作为具体证据。ResNet full的46次错误中，Speckled共有22次（8张判为Homogeneous、7张Nucleolar、6张NuMem、1张Centromere）。

ResNet34 full例子：Speckled召回率96.17%，低于该模型其他类；Golgi虽样本少，召回率98.18%。因此不能笼统说“最少样本类一定最差”。如讨论纹理相似性，应使用“可能”并说明没有专门的可解释性实验。

macro-F1可作为补充（非老师明确要求）：AlexNet partial约93.88%，AlexNet full96.43%，ResNet partial96.28%，ResNet full98.27%。不要把macro-F1与总体accuracy混为一谈。

## 5. Discussion and Conclusion（建议约20–25%正文）

### 5.1 架构的影响：为什么ResNet34 full较好？

结果：同为full，ResNet34为98.31%，AlexNet为96.62%，差1.69个百分点。结合Week11第36–37页说明残差连接有助于优化较深结构，可能更有效地学习细胞纹理与形态特征。区分观察与机制推测：本实验没有去掉残差连接的消融实验，不能把全部差异归因于残差连接。

ResNet34 full参数少于AlexNet full，因此“参数越多越好”不成立。不要声称所有深层网络都优于浅层。

### 5.2 微调策略的影响：精度与成本

- AlexNet full比partial高2.57个百分点（由未四舍五入原值计算）。
- ResNet34 full比partial高1.84个百分点。
- ImageNet自然图像与细胞图像存在域差异，全量微调允许各层适应目标域，这是合理解释，不是本实验单独证明的因果关系。
- ResNet34 partial只比AlexNet full多错4张（96.47% vs96.62%），可训练参数却少很多，是资源受限时值得考虑的折中；没有多次重复或显著性检验，不应把这4张差异夸大。
- 若资源允许，当前测试中full ResNet34准确率最高；若强调训练时间/参数预算，partial ResNet34有竞争力。不要把结果推广为所有数据集上的定论。

### 5.3 学习率、失稳与轮数

AlexNet partial、lr0.01：第20轮验证87.40%，最后约21.06%；训练accuracy也下降，loss整体升高，更像优化不稳定而非典型的“训练越来越好、验证越来越差”的过拟合。学习率过大与动量可能参与；未记录完整梯度范数/权重范数，不能唯一归因。也不能未经预测分布检查就断言始终预测某一类。

全量lr0.01发生非有限loss并被终止。固定lr没有后期衰减，可能不适合细调。可把scheduler、梯度裁剪、更小lr作为未来研究，明确此次没有使用。

60轮预算下四组最佳点为43/44/51/50；后期提升趋缓，不保证加到100轮更好。最终使用最佳验证checkpoint而非末轮，是合理的模型选择方式。

### 5.4 与课堂及论文的对应

| 来源 | 位置（PDF页序号，1起） | 可支持的内容 | 不应作出的推论 |
|---|---|---|---|
| Week11 | 21 | HEp-2分类任务背景 | 本实验等同竞赛官方评估协议 |
| Week11 | 22–25 | AlexNet、FC结构、Dropout | torchvision与经典图逐层完全相同 |
| Week11 | 26–29 | 增强、预处理、迁移与两种fine-tuning | 老师保证全量微调总是最好 |
| Week11 | 36–37 | 残差连接及plain/residual深度对比 | 更深永远更好；已证明残差是唯一原因 |
| Week11 | 38 | 另一胸片案例中ResNet50验证100%、AlexNet95% | 其数字可直接对比我们的细胞测试；ResNet50=ResNet34 |
| Week10 | 11、17、31–37 | 交叉熵/分类输出、学习率与SGD等基本原理 | 课程例子的超参数必然最适合本数据 |
| Gao等HEp-2论文 | 增强与分类方法部分 | 旋转增强动机、任务背景 | 本项目严格复现论文网络/扩增倍数/竞赛成绩 |

引用页码在最终排版前用自己的课件版本复核；Week11核心29、36–38页已经逐页检查过。

### 5.5 局限与未来工作

至少覆盖：单一固定划分和随机种子，没有重复运行置信区间；仅搜索三种lr；不同partial冻结范围意味着不能只把差异归因于架构；六类分布不均衡；数据加载策略有记录的中途升级，时长有实现开销；模型没有独立外部数据验证。样本ID不重叠不等于已验证患者/标本级独立性，没有相应元数据时不要声称完成了患者级泄漏检查。

测试集已用于最终报告，不再根据测试分数反复调整模型或轮数。未来工作可建议多随机种子、独立外部验证、学习率调度、更多冻结范围或增强消融，但不要写成已经做过。

### 5.6 结论（约一段）

概括四组比较与核心数值：全量ResNet34测试98.31%最高；partial ResNet34以更少可训练参数取得96.47%；准确率与成本需要共同衡量。说明结果适用于本次划分和实验设置，不宣称临床可用或普遍最优。

按作业文件要求，将指定词放在Section5结束处且仅一次：KAANet。（最终报告不应重复复制前面“提交提醒”中的这个词。）

## 6. References（放在Section5之后）

按同一引用格式列出原始来源。下列是核对入口，不要仅引用聊天回答：

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS. https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
3. Gao, Z., Wang, L., Zhou, L., & Zhang, J. HEp-2 Cell Image Classification with Deep Convolutional Neural Networks. 教师提供PDF首部为arXiv:1504.02531v1 (2015)。https://arxiv.org/abs/1504.02531 。若引用期刊正式版本，请另核对其年份/卷期/页码，不把预印本版本与期刊书目信息混写。
4. 课程Week10和Week11课件：补真实课程名、授课教师、年份及学校；不要编造作者信息。
5. 实现可引用torchvision文档及本项目使用的版本/权重来源，权重SHA见仓库weights/manifest.json。

## 7. 图表与附录清单

正文优先放：数据分布表、四组冻结范围/参数表、训练设置表、12候选验证表（可移附录）、最终测试+成本主表、验证曲线、选中候选train/val曲线、混淆矩阵。页面有限时，四组逐学习率详细曲线放附录。

图注模板：
- “Validation accuracy across three learning rates for each architecture and fine-tuning strategy. Stars indicate the best validation epochs. AlexNet full with lr=0.01 terminated after two completed epochs due to non-finite loss.”
- “Training and validation accuracy and cross-entropy loss for the validation-selected learning rate (0.001) of each configuration. Vertical dotted lines indicate the selected epochs. Curves show unsmoothed per-epoch values.”
- “Test-set confusion matrices for the four validation-selected models (n=2720). Rows denote ground truth and columns predictions; cell labels are counts and colors are row-normalized proportions.”

报告结尾核对：个人贡献真实；每个图表在正文引用且有图注；表内单位一致；accuracy乘100显示%；差值写“百分点”；时间口径明确；11 completed+1 diverged没有混成12个成功；没有把验证99.08%写成测试；没有把最佳epoch写成统一60；没有声称做过scheduler/early stopping/TTA/交叉验证。
