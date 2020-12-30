# /usr/bin/bash

# Copyright 2020 RUC & Johns Hopkins University (author: Shuai Guo, Jiatong Shi)

# . ./path.sh

set -e

stage=0
 

if [ $stage -le 1 ]; then
  # download the data.  
  echo $stage
fi

if [ $stage -le 2 ]; then
  # data preprocessing & format into different set(trn/val/tst)
  echo ============================================================================
  echo "       data preprocessing & format into different set(trn/val/tst)        "
  echo ============================================================================

  python local/prepare_data.py kiritan_singing/wav kiritan_singing/mono_label kiritan_data

fi

if [ $stage -le 3 ]; then
  # collect_stats
  echo ============================================================================
  echo "                          collect_stats                                   "
  echo ============================================================================

  
  cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_rnn.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_rnn_norm.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_rnn_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_rnn_norm_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_rnn_dmel_gnorm_perp.yaml"


  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_puretransformer.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_puretransformer_norm.yaml"

  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_norm.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_dmel.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_dmel_gnorm.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_perp_gnorm.yaml"
  # cmd="python ../../../SVS/bin/train.py --collect_stats=True -c conf/train_glu_dmel_gnorm_perp.yaml"
  

  echo $cmd | sh
  
fi

if [ $stage -le 4 ]; then
  # train
  echo ============================================================================
  echo "                                train                                     "
  echo ============================================================================

  
  cmd="python ../../../SVS/bin/train.py -c conf/train_rnn.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_rnn_norm.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_rnn_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_rnn_norm_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_rnn_dmel_gnorm_perp.yaml"


  # cmd="python ../../../SVS/bin/train.py -c conf/train_puretransformer.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_puretransformer_norm.yaml"
  
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_norm.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_dmel.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_perp.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_dmel_gnorm.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_perp_gnorm.yaml"
  # cmd="python ../../../SVS/bin/train.py -c conf/train_glu_dmel_gnorm_perp.yaml"

  echo $cmd | sh
fi
