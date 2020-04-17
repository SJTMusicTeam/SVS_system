#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)



import torch
import os
from model.SVSDataset import SVSDataset, SVSCollator
from model.network import GLU_TransformerSVS
from model.loss import MaskedLoss
from model.utils import AverageMeter, create_src_key_padding_mask, log_figure


def infer(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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


    # Load model weights
    print("Loading pretrained weights from {}".format(args.model_file))
    checkpoint = torch.load(args.model_file, map_location=device)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    state_dict_new = {}
    para_list = []
    for k, v in state_dict.items():
        assert k in model_dict
        if model_dict[k].size() == state_dict[k].size():
            state_dict_new[k] = v
        else:
            para_list.append(k)

    print("Total {} parameters, loaded {} parameters".format(len(state_dict), len(state_dict_new)))

    if len(para_list) > 0:
        print("Not loading {} because of different sizes".format(", ".join(para_list)))
    model_dict.update(state_dict_new)
    model.load_state_dict(model_dict)
    print("Loaded checkpoint {}".format(args.model_file))
    model = model.to(device)
    model.eval()
    

    # Decode
    test_set = SVSDataset(align_root_path=args.test_align,
                           pitch_beat_root_path=args.test_pitch,
                           wav_root_path=args.test_wav,
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
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn_svs,
                                               pin_memory=True)

    if args.loss == "l1":
        loss = MaskedLoss("l1")
    elif args.loss == "mse":
        loss = MaskedLoss("mse")
    else:
        raise ValueError("Not Support Loss Type")

    losses = AverageMeter()

    if not os.path.exists(args.prediction_path):
        os.makedirs(args.prediction_path)

    for step, (phone, beat, pitch, spec, length, chars, char_len_list) in enumerate(test_loader, 1):
        if step >= args.decode_sample:
            break
        phone = phone.to(device)
        beat = beat.to(device)
        pitch = pitch.to(device).float()
        spec = spec.to(device).float()

        chars = chars.to(device)
        length_mask = create_src_key_padding_mask(length, args.num_frames)
        length_mask = length_mask.unsqueeze(2)
        length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
        length_mask = length_mask.to(device)
        length = length.to(device)
        char_len_list = char_len_list.to(device)

        output, att = model(chars, phone, pitch, beat, src_key_padding_mask=length,
                       char_key_padding_mask=char_len_list)

        test_loss = loss(output, spec, length_mask)
        if step % 1 == 0:
        	log_figure(step, output, spec, att, length, args.prediction_path, args)
        losses.update(test_loss.item(), phone.size(0))
    print("loss avg for test is {}".format(losses.avg))
