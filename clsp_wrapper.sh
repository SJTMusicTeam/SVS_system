. ./path.sh
cuda_cmd="queue-freegpu.pl --mem 8G --gpu 1 --config conf/gpu.conf"


${cuda_cmd} --gpu 1 exp/debug_rnn/train.log train.sh
