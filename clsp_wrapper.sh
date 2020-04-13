. ./path.sh
cuda_cmd="queue-freegpu.pl --mem 4G --gpu 1 --config conf/gpu.conf"


${cuda_cmd} --gpu 1 exp/debug/train.log run.sh
