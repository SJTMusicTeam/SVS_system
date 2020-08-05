# /usr/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

# . ./path.sh

set -e

stage=3
 

if [ $stage -le 1 ]; then
  # download the data.  
  echo $stage
fi

if [ $stage -le 2 ]; then
  # data preprocessing & format into different set(trn/val/tst)
  echo ============================================================================
  echo "       data preprocessing & format into different set(trn/val/tst)        "
  echo ============================================================================

  cmd="python ../../../tools/prepare_data.py kiritan_singing/wav kiritan_singing/mono_label kiritan_data"
  echo $cmd | sh
  
fi

if [ $stage -le 3 ]; then
  # collect_stats
  echo ============================================================================
  echo "                          collect_stats                                   "
  echo ============================================================================

  cmd="python ../../../bin/train.py --collect_stats=True -c conf/train_rnn_gs.yaml"
  echo $cmd | sh
  
fi

if [ $stage -le 4 ]; then
  # train
  echo ============================================================================
  echo "                                train                                     "
  echo ============================================================================

  cmd="python ../../../bin/train.py -c conf/train_rnn_gs.yaml"
  echo $cmd | sh
fi