#!/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=1
stop_stage=2
ngpu=1
raw_data_dir=data
download_wavernn_vocoder=False
vocoder=wavernn

# model_name=conformer_full
model_name=glu
# model_name=rnn

train_config=conf/train_${model_name}.yaml
infer_config=conf/infer_${model_name}.yaml

expdir=exp/glu_norm_pe_1e-4_augment
# expdir=exp/conformer_norm_pe_1e-4
# expdir=exp/conformer_norm_pe_1e-4_augment

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Stage1: data preprocessing
  echo =============================
  echo " Stage1: data preprocessing "
  echo =============================
  mkdir -p ${raw_data_dir}
  python local/prepare_data.py data ..

  if [ ${download_wavernn_vocoder} = True ]; then
    wget -nc https://raw.githubusercontent.com/pppku/model_zoo/main/wavernn/latest_weights.pyt -P ${expdir}/model/wavernn
  fi

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
    train.py \
      --db_joint True \
       -c conf/train_${model_name}_wavernn.yaml \
      --collect_stats True \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
    else
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
    train.py \
      --db_joint True \
       -c ${train_config} \
      --collect_stats True \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Stage3: train
  echo ===============
  echo " Stage3: train "
  echo ===============

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      --db_joint True \
      -c conf/train_${model_name}_wavernn.yaml \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz \
      --vocoder_category ${vocoder} \
      --wavernn_voc_model ${expdir}/model/wavernn/latest_weights.pyt
  else
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      --db_joint True \
      -c ${train_config} \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
  fi

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Stage4: inference
  echo ===============
  echo " Stage4: infer "
  echo ===============

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
    infer.py \
      --db_joint True \
      -c conf/infer_${model_name}_wavernn.yaml \
      --prediction_path ${expdir}/infer_result \
      --model_file ${expdir}/epoch_loss_65.pth.tar \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz \
      --vocoder_category ${vocoder} \
      --wavernn_voc_model ${expdir}/model/wavernn/latest_weights.pyt
  else
    ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
    infer.py \
      --db_joint True \
      -c conf/infer_${model_name}.yaml \
      --prediction_path ${expdir}/infer_result \
      --model_file ${expdir}/epoch_loss_65.pth.tar \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
  fi

fi

