#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import os
import sys
import numpy as np
import torch
import time
from model.gpu_util import use_single_gpu
from model.SVSDataset import SVSDataset, SVSCollator
from model.network import GLU_Transformer
from model.transformer_optim import ScheduledOptim
from model.loss import MaskedLoss
from model.utils import train_one_epoch, save_checkpoint, validate, record_info


def train(args):
    if args.gpu > 0 and torch.cuda.is_available():
        cvd = use_single_gpu()
        print(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SVSDataset(align_root_path=args.train_align,
                           pitch_beat_root_path=args.train_pitch,
                           wav_root_path=args.train_wav,
                           char_max_len=args.char_max_len,
                           max_len=args.num_frames,
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
                           char_max_len=args.char_max_len,
                           max_len=args.num_frames,
                           sr=args.sampling_rate,
                           preemphasis=args.preemphasis,
                           frame_shift=args.frame_shift,
                           frame_length=args.frame_length,
                           n_mels=args.n_mels,
                           power=args.power,
                           max_db=args.max_db,
                           ref_db=args.ref_db)
    collate_fn_svs = SVSCollator(args.num_frames, args.char_max_len)
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
    # print(dev_set[0][3].shape)
    assert args.feat_dim == dev_set[0][3].shape[1]

    # prepare model
    if args.model_type == "GLU_Transformer":
        model = GLU_Transformer(phone_size=args.phone_size,
                                embed_size=args.embedding_size,
                                hidden_size=args.hidden_size,
                                glu_num_layers=args.glu_num_layers,
                                dropout=args.dropout,
                                output_dim=args.feat_dim,
                                dec_nhead=args.dec_nhead,
                                dec_num_block=args.dec_num_block)
    else:
        raise ValueError('Not Support Model Type %s' % args.model_type)
    print(model)
    model = model.to(device)

    # load weights for pre-trained model
    if args.initmodel != '':
        pretrain = torch.load(args.initmodel, map_location=device)
        pretrain_dict = pretrain['state_dict']
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        for k, v in pretrain_dict.items():
            assert k in model_dict
            if model_dict[k].size() == pretrain_dict[k].size():
                state_dict_new[k] = v
            else:
                para_list.append(k)
        print("Total {} parameters, Loaded {} parameters".format(
            len(pretrain_dict), len(state_dict_new)))
        if len(para_list) > 0:
            print("Not loading {} because of different sizes".format(
                ", ".join(para_list)))
        model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        print("Loaded checkpoint {}".format(args.initmodel))
        print("")


    # setup optimizer
    if args.optimizer == 'noam':
        optimizer = ScheduledOptim(torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-09),
            args.hidden_size,
            args.noam_warmup_steps,
            args.noam_scale)
    else:
        raise ValueError('Not Support Optimizer')

    # Setup tensorborad logger
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("{}/log".format(args.model_save_dir))
    else:
        logger = None

    if args.loss == "l1":
        loss = MaskedLoss(torch.nn.L1Loss())
    elif args.loss == "mse":
        loss = MaskedLoss(torch.nn.MSELoss())
    else:
        raise ValueError("Not Support Loss Type")

    # Training
    for epoch in range(1, 1 + args.max_epochs):
        start_t_train = time.time()
        train_info = train_one_epoch(train_loader, model, device, optimizer, loss, args)
        end_t_train = time.time()

        print(
            'Train epoch: {:04d}, lr: {:.6f}, '
            'loss: {:.4f}, time: {:.2f}s'.format(
                epoch, optimizer._optimizer.param_groups[0]['lr'],
                train_info['loss'], end_t_train - start_t_train))

        start_t_dev = time.time()
        dev_info = validate(dev_loader, model, device, loss, args)
        end_t_dev = time.time()

        print("Epoch: {:04d}, Valid loss: {:.4f}, time: {:.2f}s".format(
            epoch, dev_info['loss'], end_t_dev - start_t_dev))
        print("")
        sys.stdout.flush()
        
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer._optimizer.state_dict(),
        }, "{}/epoch_{}.pth.tar".format(args.model_save_dir, epoch))

        # record training and validation information
        if args.use_tfboard:
            record_info(train_info, dev_info, epoch, logger)

    if args.use_tfboard:
        logger.close()

