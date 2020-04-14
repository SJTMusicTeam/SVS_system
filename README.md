# SVS_system
A system works on singing voice synthesis

## Environment Install

`cd tools` \
`make KALDI=your_kaldi_dir`
and then install torch by conda

For usage: \
`. tools/env.sh`

Followings are my sample environment: \
python version: 3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0] \
pytorch version: 1.1.0 \
cupy version: 7.3.0 \
cuda version: 9000 \
cudnn version: 7501

## Running Instruction

For CLSP User, using clsp_wrapper to use qsub.

For other user, using train.sh or infer.sh to run.

Please refer to configuration file (e.g. train.yaml) for parameter.
