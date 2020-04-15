. ./path.sh
. ./cmd.sh

${cuda_cmd} --gpu 1 exp/debug/train.log train.sh
