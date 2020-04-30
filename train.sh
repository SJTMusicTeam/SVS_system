# /usr/bin/bash

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

. ./path.sh

CUDA_VISIBLE_DEVICES=1, python bin/train.py -c conf/train_transformer_glu.yaml

