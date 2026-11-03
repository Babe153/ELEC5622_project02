# ELEC5622 Project 02

HEp-2 cell image classification using ImageNet-pretrained AlexNet and ResNet34.
Compare partial and full fine-tuning with three learning rates per configuration.

**Status: code and pretrained-weight loading checked; formal training has not started.**

## Setup

Python 3.10 and a compatible NVIDIA GPU environment are recommended.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_weights.py
```

Place the instructor-provided data in this layout (data are not included):

```text
datasets/
  training/
  validation/
  test/
gt_training.csv
```

## Commands

```powershell
.\run.ps1 -Action check
# Only when ready to start training:
.\run.ps1 -Action train -Experiment baseline_30epochs
# After completing all validation-based model selection:
.\run.ps1 -Action test -Experiment baseline_30epochs
```

The launcher uses `.venv/Scripts/python.exe` by default. To use another environment,
copy `local_environment.example.json` to `local_environment.json` and set its Python path.
The machine-specific file is ignored by Git.

## Experiment design

- AlexNet: partial tuning of FC6-FC8, or full tuning.
- ResNet34: partial tuning of layer4 and fc, or full tuning.
- Learning rates: 0.01, 0.001, 0.0001; maximum 30 epochs each (12 trials).
- SGD, momentum 0.9, weight decay 0.0005, batch size 16.
- 224x224 RGB input, online rotation, ImageNet normalization, CUDA mixed precision.
- Validation accuracy selects checkpoints; validation loss breaks ties.
- The test set is evaluated only through the separate test command.

See [中文使用说明](README_中文.md) for details, timing definitions and limitations.
Datasets, pretrained binaries, model outputs, source PDF and local environment settings
are intentionally excluded. `download_weights.py` retrieves official weights and verifies checksums.
