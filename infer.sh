# /usr/bin/bash

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

. ./path.sh

# CUDA_VISIBLE_DEVICES=0

python bin/infer.py -c conf/infer.yaml
