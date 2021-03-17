#! /usr/bin/bash

# Copyright 2020 RUC & Johns Hopkins University (author: Shuai Guo, Jiatong Shi)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=100
ngpu=1
raw_data_dir=downloads
expdir=exp/rnn_norm_perp

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
  # Stage0: download data
  echo =======================
  echo " Stage0: download data "
  echo =======================
  mkdir -p ${raw_data_dir}
  echo "please download kiritan dataset from https://zunko.jp/kiridev/login.php, requires a Facebook account due to licensing issues"
  echo "put kiritan_singing.zip under ${raw_data_dir}"
  unzip -o ${raw_data_dir}/kiritan_singing.zip -d ${raw_data_dir}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
  # Stage1: data preprocessing & format into different set(trn/val/tst)
  echo ============================
  echo " Stage1: data preprocessing "
  echo ============================

  python local/prepare_data.py ${raw_data_dir}/kiritan_singing/wav ${raw_data_dir}/kiritan_singing/mono_label ${raw_data_dir}/kiritan_data --use_pyworld_vocoder Ture
  ./local/train_dev_test_split.sh data train dev test
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train.py \
    -c conf/train_conformer_gnorm_perp.yaml \
    --collect_stats True \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz 
  
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then 
  # Stage3: train
  echo ===============
  echo " Stage3: train "
  echo ===============

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train.py \
    -c conf/train_conformer_gnorm_perp.yaml \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then 
  # Stage4: inference
  echo ===============
  echo " Stage4: infer "
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
  infer.py -c conf/infer.yaml


fi
