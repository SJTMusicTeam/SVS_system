#!/bin/bash


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
#utils/fix_data_dir.sh

data=data

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
echo ============================================================================
echo "                MFCC Extraction                     "
echo ============================================================================

mfccdir=mfcc_svs
for x in clean_set other_set; do
  utils/fix_data_dir.sh $data/$x
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 32 $data/$x exp/make_mfcc_svs/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh $data/$x exp/make_mfcc_svs/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh $data/$x || exit 1;
done
exit 1;
echo ============================================================================
echo "                Decoding                     "
echo ============================================================================

#utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 32 --config conf/decode.config \
   exp/tri5a/graph $data/clean_set exp/tri5a/decode_svs_clean_set || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 32 --config conf/decode.config \
   exp/tri5a/graph $data/other_set exp/tri5a/decode_svs_other_set || exit 1;


