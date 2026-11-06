| model | params | trainable_params | input | accuracy | precision | recall | f1 | roc_auc | epochs | train_min | trained_on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 4,287,809 | 4,287,809 | 128×128 | 0.8458 | 0.8362 | 0.8592 | 0.8476 | 0.9185 | 10 | 23.6 | CPU |
| improved_cnn | 4,288,865 | 4,288,161 | 128×128 | 0.8573 | 0.8548 | 0.8603 | 0.8575 | 0.9341 | 26 | 21.6 | GPU (RTX 5060) |
| mobilenetv2 | 2,259,265 | 1,512,001 | 160×160 | 0.9892 | 0.9882 | 0.9903 | 0.9892 | 0.9992 | 17 | 23.7 | GPU (RTX 5060) |
