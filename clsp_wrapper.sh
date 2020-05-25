. ./path.sh
cuda_cmd="queue-freegpu.pl --mem 8G --gpu 1 --config conf/gpu.conf"


${cuda_cmd} --gpu 1 exp/kiritan_gaussian_mask_glu/train.log train.sh
