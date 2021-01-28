#!/bin/bash

# Copyright 2020 RUC & Johns Hopkins University (author: Shuai Guo, Jiatong Shi, Lei Li)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=3
ngpu=1
raw_data_dir=downloads
expdir=exp/rnn

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
  ./local/download_and_untar.sh ${raw_data_dir} 'https://uc3ac3d286cfd5e74fddc5e89b2f.dl.dropboxusercontent.com/cd/0/get/BH0ymVh2ZYSeU77SkE3bKVE3U7677EXiYe9RAg9JbHcCEKPmr8b86uN74Mb2aqwE8CSA_fCieF4kXgxtAYJNPIn-8cbcDL3BW0SUqikk7HHDBEEu_I5ktIyM0kLuOQe7vys/file?_download_id=55245822596510105054397496268015654120582580896732133304782966454&_notify_doma' OFUTON_P_UTAGOE_DB.zip
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
  # Stage1: data preprocessing & format into different set(trn/val/tst)
  echo ============================
  echo " Stage1: data preprocessing "
  echo ============================

  python local/prepare_data.py ${raw_data_dir}/OFUTON_P_UTAGOE_DB/wav ${raw_data_dir}/OFUTON_P_UTAGOE_DB/lab data \
    --label_type r --wav_extention wav
  ./local/train_dev_test_split.sh data train dev test
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train.py \
    -c conf/train_rnn.yaml \
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

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
  train.py \
    -c conf/train_rnn.yaml \
    --gpu_id -1 \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then 
  # Stage4: inference
  echo ===============
  echo " Stage3: infer "
  echo ===============

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_infer.log \
  infer.py -c conf/infer_rnn.yaml

fi

