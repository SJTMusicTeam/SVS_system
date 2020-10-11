#!/bin/bash


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=1
nj=10
model=TDNN # or HMM is end with HMM training
workspace="aishell_try/"
infer_set="clean_set other_set"

if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "               Data Preparation                     "
  echo ============================================================================
  ./local/data_prep.sh || exit 1;
  for datadir in ${infer_set}; do
    utils/fix_data_dir.sh data/${datadir}
  done
fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

if [ $model == "TDNN" ] &&[ $stage -le 2 ]; then
  echo ============================================================================
  echo "                MFCC Extraction                     "
  echo ============================================================================
  mfccdir=mfcc_hires
  for datadir in ${infer_set}; do
    utils/copy_data_dir.sh data/${datadir} data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires || exit 1;
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
        --nj $nj data/${datadir}_hires exp/make_mfcc/ ${mfccdir}
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_mfcc ${mfccdir}
    utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires data/${datadir}_hires_nopitch
    steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_mfcc ${mfccdir}
  done
  
  for datadir in ${infer_set}; do
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${datadir}_hires_nopitch data/${datadir}_hires_nopitch_max2
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${datadir}_hires_nopitch_max2 ${workspace}exp/nnet3/extractor ${workspace}exp/nnet3/ivectors_${datadir} || exit 1;
  done
  
fi

if [ $model == "HMM" ] && [ ${stage} -le 2 ]; then
  # HMM version inference
  mfccdir=mfcc_svs
  for x in clean_set other_set; do
   utils/fix_data_dir.sh $data/$x
   steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 32 $data/$x exp/make_mfcc_svs/$x $mfccdir || exit 1;
   steps/compute_cmvn_stats.sh $data/$x exp/make_mfcc_svs/$x $mfccdir || exit 1;
   utils/fix_data_dir.sh $data/$x || exit 1;
  done
fi

if [ $model == "TDNN" ] && [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                Decoding                     "
  echo ============================================================================
  
  for datadir in ${infer_set}; do
   steps/chain/get_phone_post.sh --nj $nj --remove-word-position-dependency false \
      --online-ivector-dir aishell_try/exp/nnet3/ivectors_${datadir} \
      ${workspace}exp/chain/tri6_7d_tree_sp/ ${workspace}exp/chain/tdnn_1a_sp ${workspace}data/lang_test data/${datadir}_hires exp/chain/tdnn_${datadir}
    for jbs in $(seq 1 ${nj}); do
      copy-feats ark:exp/chain/tdnn_${datadir}/phone_post.${jbs}.ark ark,t:exp/chain/tdnn_${datadir}/phone_post.${jbs}.txt
    done
  done
fi

if [ $model == "HMM" ] && [ ${stage} -le 3]; then
  echo ============================================================================
  echo "                Decoding                     "
  echo ============================================================================
  
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 32 --config conf/decode.config \
     ${workspace}exp/tri5a/graph data/clean_set ${workspace}exp/tri5a/decode_svs_clean_set || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 32 --config conf/decode.config \
     ${workspace}exp/tri5a/graph data/other_set ${workspace}exp/tri5a/decode_svs_other_set || exit 1;

  echo ============================================================================
  echo "               Phone posterior probability                     "
  echo ============================================================================
  
  root_4=${workspace}exp/tri5a
  mkdir phone_post
  mkdir phone_post/clean
  mkdir phone_post/other
  for dir in decode_svs_clean_set decode_svs_other_set; do
    for x in $root_4/$dir/lat.*.gz; do
      cp $x phone_post/clean
      gzip -d $x
    done
  done

  for dir in clean other; do
    for x in phone_post/$dir/lat.*; do
      #gzip -d $x
      num=${x##*.}
      lattice-to-post ark:$x ark,t:phone_post/$dir/$num.post
      post-to-phone-post $root_4/35.mdl ark:phone_post/$dir/$num.post ark,t:phone_post/$dir/phone_$num.post
    done
  done

fi

