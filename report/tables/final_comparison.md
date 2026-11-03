# Verified final comparison

Accuracy is a fraction in CSV and a percentage below.

| Configuration | Best epoch | Validation | Test | Errors | Parameters | Training to best (min) | Training 60 epochs (min) | All-LR tuning wall time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alexnet_partial | 43 | 95.91% | 94.04% | 162 | 54,558,726 | 11.87 | 16.57 | 61.24 |
| alexnet_full | 44 | 97.79% | 96.62% | 92 | 57,028,422 | 15.33 | 20.77 | 48.57 |
| resnet34_partial | 51 | 97.52% | 96.47% | 96 | 13,117,446 | 15.76 | 18.51 | 67.07 |
| resnet34_full | 50 | 99.08% | 98.31% | 46 | 21,287,750 | 33.72 | 40.40 | 130.26 |
