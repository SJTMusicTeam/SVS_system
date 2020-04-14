#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)



import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from librosa.output import write_wav
from librosa.display import specshow
from model.SVSDataset import SVSDataset, SVSCollator
from model.network import GLU_Transformer
from model.loss import MaskedLoss
from model.utils import AverageMeter, spectrogram2wav, create_src_key_padding_mask


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

    for step, (phone, beat, pitch, spec, real, imag, length, chars, char_len_list) in enumerate(test_loader, 1):
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
            # save wav and plot spectrogram
            output = output.cpu().detach().numpy()[0]
            out_spec = spec.cpu().detach().numpy()[0]
            length = length.cpu().detach().numpy()[0]
            att = att.cpu().detach().numpy()[0]
            # np.save("output.npy", output)
            # np.save("out_spec.npy", out_spec)
            # np.save("att.npy", att)
            output = output[:length]
            out_spec = out_spec[:length]
            att = att[:, :length, :length]
            wav = spectrogram2wav(output, args.max_db, args.ref_db, args.preemphasis, args.power, args.sampling_rate, args.frame_shift, args.frame_length)
            wav_true = spectrogram2wav(out_spec, args.max_db, args.ref_db, args.preemphasis, args.power, args.sampling_rate, args.frame_shift, args.frame_length)
            write_wav(os.path.join(args.prediction_path, '{}.wav'.format(step)), wav, args.sampling_rate)
            write_wav(os.path.join(args.prediction_path, '{}_true.wav'.format(step)), wav_true, args.sampling_rate)
            plt.subplot(1, 2, 1)
            specshow(output.T)
            plt.title("prediction")
            plt.subplot(1, 2, 2)
            specshow(out_spec.T)
            plt.title("ground_truth")
            plt.savefig(os.path.join(args.prediction_path, '{}.png'.format(step)))
            plt.subplot(1, 4, 1)
            specshow(att[0])
            plt.subplot(1, 4, 2)
            specshow(att[1])
            plt.subplot(1, 4, 3)
            specshow(att[2])
            plt.subplot(1, 4, 4)
            specshow(att[3])
            plt.savefig(os.path.join(args.prediction_path, '{}_att.png'.format(step)))
        losses.update(test_loss.item(), phone.size(0))
    print("loss avg for test is {}".format(losses.avg))
