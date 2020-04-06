#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import yamlargparse
import os
import sys
import numpy as np
import torch
import time
import subprocess
from argparse import ArgumentParser
from model.gpu_util import use_single_gpu
from model.SVSDataset import SVSDataset, SVSCollator


def train(args):
    if args.gpu > 0 and torch.cuda.is_available():
        cvd = use_single_gpu()
        print(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set = SVSDataset(align_root_path=args.train_align,
                           pitch_beat_root_path=args.train_pitch,
                           wav_root_path=args.train_wav,
                           sr=args.sampling_rate,
                           preemphasis=args.preemphasis,
                           frame_shift=args.frame_shift,
                           frame_length=args.frame_length,
                           n_mels=args.n_mels,
                           power=args.power,
                           max_db=args.max_db,
                           ref_db=args.ref_db)

    dev_set = SVSDataset(align_root_path=args.val_align,
                           pitch_beat_root_path=args.val_pitch,
                           wav_root_path=args.val_wav,
                           sr=args.sampling_rate,
                           preemphasis=args.preemphasis,
                           frame_shift=args.frame_shift,
                           frame_length=args.frame_length,
                           n_mels=args.n_mels,
                           power=args.power,
                           max_db=args.max_db,
                           ref_db=args.ref_db)
    collate_fn_svs = SVSCollator(args.num_frames)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn_svs,
                                               pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn_svs,
                                               pin_memory=True)
    assert args.feat_dim == dev_set[0][3].shape[1]

    # prepare model
    if args.model_type == "GLU_Transformer"
        model = GLU_Transformer()
