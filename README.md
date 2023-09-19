# Disclaimer
This is not an officially supported Tencent product.

# Overview
This repository contains the code for FOLNet (First-Order Logic Network), including the model implementation, the pretraining pipeline and the finetuning pipeline. The full paper on FOLNet is accepted to ICLR2023 with openreview link "[Learning Language Representations with Logical Inductive Bias](https://openreview.net/forum?id=rGeZuBRahju)".

# Docker environment setup
The code can be run by using the docker environment, where the docker image can be built by using the dockerfile in the directory `docker`:

`docker build -t folnet .`

# Usage
## Pretraining
Run pretraining using by using the script `pretrain_FOLNet.sh`.

## Finetuning
1. Finetuning on GLUE: `finetune_FOLNet_GLUE.sh`
2. Finetuning on SQuADv2: `finetune_FOLNet_SQuADv2.sh`

# Model checkpoint
- The model checkpoints can be downloaded from:
[FOLNet-checkpoints](https://tencentoverseas-my.sharepoint.com/:f:/g/personal/jianshuchen_global_tencent_com/Em7QLOIa6bZGuqLoqylyow4BZrL-k3ZiWysXE7tiyyAxjA?e=bnAeH9) or [FOLNet-checkpoints2](https://tencentoverseas-my.sharepoint.com/:f:/g/personal/xiaomanpan_global_tencent_com/EvRCMNsu1NRAlF7Uestta54BeKWJhDk_Gw_RNzdcMAwgSw?e=Kfi4Zw)
- You need to have both the `*.cfg` file (model configuration) and the `*.pt` file (model file).

# Reference
Please cite the paper in the following format if you use this model during your research.

```
@inproceedings{chen2022learning,
  title={Learning Language Representations with Logical Inductive Bias},
  author={Chen, Jianshu},
  booktitle={International Conference on Learning Representations (ICLR)},
  address={Kigali, Rwanda}
  month={May},
  year={2023}
}
```
