#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi & Shuai Guo)


import os
import sys
import numpy as np
import torch
import time
from SVS.model.utils.gpu_util import use_single_gpu
from SVS.model.utils.SVSDataset import SVSDataset, SVSCollator
from SVS.model.network import GLU_TransformerSVS,LSTMSVS, GRUSVS_gs, TransformerSVS
from SVS.model.utils.transformer_optim import ScheduledOptim
from SVS.model.utils.loss import MaskedLoss, cal_spread_function, cal_psd2bark_dict, PerceptualEntropy
from SVS.model.utils.utils import train_one_epoch, save_checkpoint, validate, record_info, collect_stats, save_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    if args.gpu > 0 and torch.cuda.is_available() and args.auto_select_gpu == True:
        cvd = use_single_gpu()
        print(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
    else:
        torch.cuda.set_device(args.gpu_id)
        print(f"GPU {args.gpu_id} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

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
                           nfft=args.nfft,
                           frame_shift=args.frame_shift,
                           frame_length=args.frame_length,
                           n_mels=args.n_mels,
                           power=args.power,
                           max_db=args.max_db,
                           ref_db=args.ref_db,
                           sing_quality=args.sing_quality,
                           standard=args.standard)

    dev_set = SVSDataset(align_root_path=args.val_align,
                           pitch_beat_root_path=args.val_pitch,
                           wav_root_path=args.val_wav,
                           char_max_len=args.char_max_len,
                           max_len=args.num_frames,
                           sr=args.sampling_rate,
                           preemphasis=args.preemphasis,
                           nfft=args.nfft,
                           frame_shift=args.frame_shift,
                           frame_length=args.frame_length,
                           n_mels=args.n_mels,
                           power=args.power,
                           max_db=args.max_db,
                           ref_db=args.ref_db,
                           sing_quality=args.sing_quality,
                           standard=args.standard)
    collate_fn_svs = SVSCollator(args.num_frames, args.char_max_len, args.use_asr_post, args.phone_size, args.n_mels)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn_svs,
                                               pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                               batch_size=args.batchsize,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn_svs,
                                               pin_memory=True)
    # print(dev_set[0][3].shape)
    assert args.feat_dim == dev_set[0][3].shape[1]
    if args.collect_stats:
        collect_stats(train_loader,args)
        print(f"collect_stats finished !")
        quit()
    # prepare model
    if args.model_type == "GLU_Transformer":
        model = GLU_TransformerSVS(phone_size=args.phone_size,
                                embed_size=args.embedding_size,
                                hidden_size=args.hidden_size,
                                glu_num_layers=args.glu_num_layers,
                                dropout=args.dropout,
                                output_dim=args.feat_dim,
                                dec_nhead=args.dec_nhead,
                                dec_num_block=args.dec_num_block,
                                n_mels=args.n_mels,
                                double_mel_loss=args.double_mel_loss,
                                local_gaussian=args.local_gaussian,
                                device=device)
    elif args.model_type == "LSTM":
        model = LSTMSVS(phone_size=args.phone_size,
                        embed_size=args.embedding_size,
                        d_model=args.hidden_size,
                        num_layers=args.num_rnn_layers,
                        dropout=args.dropout,
                        d_output=args.feat_dim,
                        n_mels=args.n_mels,
                        device=device,
                        use_asr_post=args.use_asr_post)
    elif args.model_type == "GRU_gs":
        model = GRUSVS_gs(phone_size=args.phone_size,
                        embed_size=args.embedding_size,
                        d_model=args.hidden_size,
                        num_layers=args.num_rnn_layers,
                        dropout=args.dropout,
                        d_output=args.feat_dim,
                        n_mels=args.n_mels,
                        device=device,
                        use_asr_post=args.use_asr_post)
    elif args.model_type == "PureTransformer":
        model = TransformerSVS(phone_size=args.phone_size,
                                        embed_size=args.embedding_size,
                                        hidden_size=args.hidden_size,
                                        glu_num_layers=args.glu_num_layers,
                                        dropout=args.dropout,
                                        output_dim=args.feat_dim,
                                        dec_nhead=args.dec_nhead,
                                        dec_num_block=args.dec_num_block,
                                        n_mels=args.n_mels,
                                        double_mel_loss=args.double_mel_loss,
                                        local_gaussian=args.local_gaussian,
                                        device=device)
    
    elif args.model_type == "USTC_DAR":
        waiting_finish = 1
    

    else:
        raise ValueError('Not Support Model Type %s' % args.model_type)
    print(model)
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model_load_dir = ""
    pretrain_encoder_dir = ""
    start_epoch = 1                         # FIX ME
    if args.pretrain_encoder != '':
        pretrain_encoder_dir = args.pretrain_encoder
    if args.initmodel != '':
        model_load_dir = args.initmodel
    if args.resume and os.path.exists(args.model_save_dir):
        checks = os.listdir(args.model_save_dir)
        start_epoch = max(list(map(lambda x: int(x[6:-8]) if x.endswith("pth.tar") else -1, checks)))
        if start_epoch < 0:
            model_load_dir = ""
        else:
            model_load_dir = "{}/epoch_{}.pth.tar".format(args.model_save_dir, start_epoch)
        
    # load encoder parm from Transformer-TTS
    if pretrain_encoder_dir != '':
        pretrain = torch.load(pretrain_encoder_dir, map_location=device)
        pretrain_dict = pretrain['model']
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        i=0
        for k, v in pretrain_dict.items():
            k_new = k[7:]
            if k_new in model_dict and model_dict[k_new].size() == pretrain_dict[k].size():
                i += 1
                state_dict_new[k_new] = v
            model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        print(f"Load {i} layers total. Load pretrain encoder success !")
    
    # load weights for pre-trained model
    if model_load_dir != '':
        print("Model Start to Load, dir: {}".format(model_load_dir))
        model_load = torch.load(model_load_dir, map_location=device)
        loading_dict = model_load['state_dict']
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        for k, v in loading_dict.items():
            # assert k in model_dict
            if k == "normalizer.mean" or k == "normalizer.std" or k == "mel_normalizer.mean" or k == "mel_normalizer.std":
                continue
            if model_dict[k].size() == loading_dict[k].size():
                state_dict_new[k] = v
            else:
                para_list.append(k)
        print("Total {} parameters, Loaded {} parameters".format(
            len(loading_dict), len(state_dict_new)))
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
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-09)
    else:
        raise ValueError('Not Support Optimizer')

    # Setup tensorborad logger
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("{}/log".format(args.model_save_dir))
    else:
        logger = None

    if args.loss == "l1":
        loss = MaskedLoss("l1", mask_free=args.mask_free)
    elif args.loss == "mse":
        loss = MaskedLoss("mse", mask_free=args.mask_free)
    else:
        raise ValueError("Not Support Loss Type")

    if args.perceptual_loss > 0:
        win_length = int(args.sampling_rate * args.frame_length)
        psd_dict, bark_num = cal_psd2bark_dict(fs=args.sampling_rate, win_len=win_length)
        sf = cal_spread_function(bark_num)
        loss_perceptual_entropy = PerceptualEntropy(bark_num, sf, args.sampling_rate, win_length, psd_dict)
    else:
        loss_perceptual_entropy = None

    # Training
    epoch_to_save = {}
    counter = 0 
    # args.num_saved_model = 5

    for epoch in range(start_epoch + 1, 1 + args.max_epochs):
        '''
        ### Train Stage
        '''
        start_t_train = time.time()
        train_info = train_one_epoch(train_loader, model, device, optimizer, loss, loss_perceptual_entropy,
                                     epoch, args)
        end_t_train = time.time()

        out_log = 'Train epoch: {:04d}, '.format(epoch)
        if args.optimizer == "noam":
            out_log += 'lr: {:.6f}, '.format(optimizer._optimizer.param_groups[0]['lr'])
        elif args.optimizer == "adam":
            out_log += 'lr: {:.6f}, '.format(optimizer.param_groups[0]["lr"])
        out_log += 'loss: {:.4f}, spec_loss: {:.4f} '.format(train_info['loss'], train_info['spec_loss'])
        
        if args.n_mels > 0:
            out_log += 'mel_loss: {:.4f}, '.format(train_info['mel_loss'])
        if args.perceptual_loss > 0:
            out_log += 'pe_loss: {:.4f}, '.format(train_info['pe_loss'])
        print("{} time: {:.2f}s".format(out_log, end_t_train - start_t_train))

        
        '''
        ### Dev Stage
        '''
        torch.backends.cudnn.enabled = False    # 莫名的bug，关掉才可以跑

        start_t_dev = time.time()
        dev_info = validate(dev_loader, model, device, loss, loss_perceptual_entropy, epoch, args)
        end_t_dev = time.time()

        dev_log = 'Dev epoch: {:04d}, loss: {:.4f}, spec_loss: {:.4f}, '.format(epoch, 
                                                                            dev_info['loss'],
                                                                            dev_info['spec_loss'])
        dev_log += 'mcd_value: {:.4f}, '.format(dev_info['mcd_value'])
        if args.n_mels > 0:
            dev_log += 'mel_loss: {:.4f}, '.format(dev_info['mel_loss'])
        if args.perceptual_loss > 0:
            dev_log += 'pe_loss: {:.4f}, '.format(dev_info['pe_loss'])
        print("{} time: {:.2f}s".format(dev_log, end_t_dev - start_t_train))

        print("")
        sys.stdout.flush()

        torch.backends.cudnn.enabled = True

        '''
        ### Save model Stage
        '''
        # if not os.path.exists(args.model_save_dir):
        #     os.makedirs(args.model_save_dir)

        # if counter < args.num_saved_model:
        #     counter += 1 
        #     if dev_info['spec_loss'] in epoch_to_save.keys():
        #         counter -= 1
        #         continue
        #     epoch_to_save[dev_info['spec_loss']] = epoch
        #     save_model(args, epoch, model, optimizer, train_info, dev_info, logger)

        # else: 
        #     sorted_dict_keys = sorted(epoch_to_save.keys(), reverse=True)
        #     sp_loss = sorted_dict_keys[0] # biggest spec_loss of saved models
        #     if dev_info['spec_loss'] < sp_loss:
        #         epoch_to_save[dev_info['spec_loss']] = epoch
        #         print('---------------------------------------------------------------------------------')
        #         print('add epoch: {:04d}, sp_loss={:.4f}'.format(epoch, dev_info['spec_loss']))

        #         if os.path.exists("{}/epoch_{}.pth.tar".format(args.model_save_dir, epoch_to_save[sp_loss])):
        #             os.remove("{}/epoch_{}.pth.tar".format(args.model_save_dir, epoch_to_save[sp_loss]))
        #             print('model of epoch:{} deleted'.format(epoch_to_save[sp_loss]))

        #         print('delete epoch: {:04d}, sp_loss={:.4f}'.format(epoch_to_save[sp_loss], sp_loss))
        #         epoch_to_save.pop(sp_loss)
                
        #         save_model(args, epoch, model, optimizer, train_info, dev_info, logger)

        #         print(epoch_to_save)
        #         print('*********************************************************************************')

        # if len(sorted(epoch_to_save.keys())) > args.num_saved_model:
        #     raise ValueError("")
    if args.use_tfboard:
        logger.close()

