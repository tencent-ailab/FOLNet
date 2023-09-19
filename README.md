# Disclaimer
This is not an officially supported Tencent product.

# Overview
This repository contains the code for FOLNet, including the model
implementation, the pretraining pipeline and the finetuning pipeline.

# Docker environment setup
The code can be run by using the docker environment, where the docker image can be built by using the dockerfile in the directory `docker`:

`docker build -t folnet .`

# Usage
## Pretraining
Run pretraining using by using the script `pretrain_FOLNet.sh`.

## Finetuning
1. Finetuning on GLUE: `finetune_FOLNet_GLUE.sh`
2. Finetuning on SQuADv2: `finetune_FOLNet_SQuADv2.sh`
3. Finetuning on FOLIO: `finetune_FOLNet_FOLIO.sh`
4. Finetuning on CLUTRR: `finetune_FOLNet_CLUTRR.sh`
