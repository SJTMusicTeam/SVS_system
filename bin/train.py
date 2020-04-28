#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import yamlargparse

import sys
sys.path.append("/home/yzhan/SVS_github/SVS_system")

parser = yamlargparse.ArgumentParser(description='SVS training')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('--train_align',
                    help='alignment data dir used for training.')
parser.add_argument('--train_pitch',
                    help='pitch data dir used for training.')
parser.add_argument('--train_wav',
                    help="wave data dir used for training")
parser.add_argument('--val_align',
                    help='alignment data dir used for validation.')
parser.add_argument('--val_pitch',
                    help='pitch data dir used for validation.')
parser.add_argument('--val_wav',
                    help="wave data dir used for validation")
parser.add_argument('--model-save-dir',
                    help='output directory which model file will be saved in.')
parser.add_argument('--model-type', default='GLU_Transformer',
                    help='Type of model (New_Transformer or GLU_Transformer or LSTM)')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--pretrain-encoder',default='',)
parser.add_argument('--resume', type=bool, default=False,
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--max-epochs', default=20, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--optimizer', default='noam', type=str)
parser.add_argument('--gradclip', default=-1, type=int,
                    help='gradient clipping. if < 0, no clipping')
parser.add_argument('--num-frames', default=100, type=int,
                    help='number of frames in one utterance')
parser.add_argument('--char_max_len', default=500, type=int,
                    help='max length for character')
parser.add_argument('--batchsize', default=1, type=int,
                    help='number of utterances in one batch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of cpu workers')
parser.add_argument('--frame-length', default=0.06, type=float)
parser.add_argument('--frame-shift', default=0.03, type=float)
parser.add_argument('--sampling-rate', default=44100, type=int)
parser.add_argument('--preemphasis', default=0.97, type=float)
parser.add_argument('--n_mels', default=80, type=int)
parser.add_argument('--power', default=1.2, type=float)
parser.add_argument('--max_db', default=100, type=int)
parser.add_argument('--ref_db', default=20, type=int)
parser.add_argument('--nfft', default=2048, type=int)
parser.add_argument('--phone_size', default=67, type=int)
parser.add_argument('--feat-dim', default=1324, type=int)
parser.add_argument('--embedding-size', default=256, type=int)
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--glu-num-layers', default=1, type=int,
                    help='number of glu layers')
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--dec_num_block', default=6, type=int)
parser.add_argument('--num-rnn-layers', default=2, type=int)
parser.add_argument('--dec_nhead', default=4, type=int)
parser.add_argument('--local_gaussian', default=False, type=bool)
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--use_tfb', dest='use_tfboard',
                    help='whether use tensorboard',
                    action='store_true')
parser.add_argument('--noam-scale', default=1.0, type=float)
parser.add_argument('--noam-warmup-steps', default=25000, type=float)
parser.add_argument('--loss', default="l1", type=str)
parser.add_argument('--perceptual_loss', default=-1, type=float)
parser.add_argument('--use-pos-enc', default=0, type=int)
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('--mask_free', default=False, type=bool)
parser.add_argument('--use_asr_post', default=False, type=bool)
parser.add_argument('--sing_quality', default="conf/sing_quality.csv", type=str)
parser.add_argument('--standard', default=3, type=int)
parser.add_argument('--train_step_log', default=100, type=int)
parser.add_argument('--dev_step_log', default=10, type=int)

parser.add_argument('--normalize',default=False,type=bool)
args = parser.parse_args()

import system_info
system_info.print_system_info()

print(args)
from model.train import train
train(args)
