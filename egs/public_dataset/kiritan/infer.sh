# /usr/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

# . ./path.sh

set -e

# infer
echo ============================================================================
echo "                                infer                                     "
echo ============================================================================


# cmd="python ../../../SVS/bin/infer.py -c conf/infer_rnn.yaml"
# cmd="python ../../../SVS/bin/infer.py -c conf/infer_rnn_perp.yaml"
cmd="python ../../../SVS/bin/infer.py -c conf/infer_rnn_norm.yaml"


# cmd="python ../../../SVS/bin/infer.py -c conf/infer_puretransformer.yaml"
# cmd="python ../../../SVS/bin/infer.py -c conf/infer_puretransformer_norm.yaml"
# cmd="python ../../../SVS/bin/infer.py -c conf/infer_glu.yaml"
# cmd="python ../../../SVS/bin/infer.py -c conf/infer_glu_dmel_gnorm.yaml"
# cmd="python ../../../SVS/bin/infer.py -c conf/infer_glu_perp.yaml"

echo $cmd | sh
