#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import os
import torch
import numpy as np
import copy
import time
import librosa
import matplotlib.pyplot as plt
from librosa.output import write_wav
from librosa.display import specshow
from scipy import signal

from pathlib import Path
from model.utterance_mvn import UtteranceMVN
# from model.global_mvn import GlobalMVN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_stats(train_loader,args):
    count,sum,sum_square=0,0,0
    count_mel,sum_mel,sum_square_mel = 0,0,0
    for step, (phone,beat,pitch,spec,real,imag,length,chars,char_len_list,mel) in enumerate(train_loader,1):
        #print(f"spec.shape: {spec.shape},length.shape: {length.shape}, mel.shape: {mel.shape}")
        for i,seq in enumerate(spec.cpu().numpy()):
            #print(f"seq.shape: {seq.shape}")
            seq_length = torch.max(length[i])
            #print(seq_length)
            seq = seq[:seq_length]
            sum += seq.sum(0)
            sum_square += (seq ** 2).sum(0)
            count += len(seq)

        for i,seq in enumerate(mel.cpu().numpy()):
            seq_length = torch.max(length[i])
            seq = seq[:seq_length]
            sum_mel += seq.sum(0)
            sum_square_mel += (seq ** 2).sum(0)
            count_mel += len(seq)
    assert count_mel == count
    np.savez(Path(args.model_save_dir) / f"feats_stats.npz",
             count=count,
             sum=sum,
             sum_square=sum_square)
    np.savez(Path(args.model_save_dir) / f"feats_mel_stats.npz",
            count = count_mel,
            sum = sum_mel,
            sum_square = sum_square_mel)
    


def train_one_epoch(train_loader, model, device, optimizer, criterion, perceptual_entropy, epoch, args):
    losses = AverageMeter()
    spec_losses = AverageMeter()
    if args.perceptual_loss > 0:
        pe_losses = AverageMeter()
    if args.n_mels > 0:
        mel_losses = AverageMeter()
    model.train()

    log_save_dir = os.path.join(args.model_save_dir, "epoch{}/log_train_figure".format(epoch))
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    start = time.time()
    for step, (phone, beat, pitch, spec, real, imag, length, chars, char_len_list, mel) in enumerate(train_loader, 1):
        phone = phone.to(device)
        beat = beat.to(device)
        pitch = pitch.to(device).float()
        spec = spec.to(device).float()
        mel = mel.to(device).float()
        real = real.to(device).float()
        imag = imag.to(device).float()
        length_mask = length.unsqueeze(2)
        length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
        length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
        length_mask = length_mask.to(device)
        length_mel_mask = length_mel_mask.to(device)
        length = length.to(device)
        char_len_list = char_len_list.to(device)

        if not args.use_asr_post:
            chars = chars.to(device)
            char_len_list = char_len_list.to(device)
        else:
            phone = phone.float()
        
        if args.model_type == "GLU_Transformer":
            output, att, output_mel = model(chars, phone, pitch, beat, pos_char=char_len_list,
                       pos_spec=length)
        elif args.model_type == "LSTM":
            output, hidden, output_mel = model(phone, pitch, beat)
            att = None
        elif args.model_type == "PureTransformer":
            output, att, output_mel = model(chars, phone, pitch, beat, pos_char=char_len_list,
                       pos_spec=length)
        elif args.model_type == "PureTransformer_norm":
            output,att,output_mel,spec,mel = model(spec,mel,chars,phone,pitch,beat,\
                    pos_char=char_len_list,pos_spec=length) # this model for global norm 

        if args.normalize:
            normalizer = UtteranceMVN()
            spec,_ = normalizer(spec,length)
            mel,_ = normalizer(mel,length)
        
        
        spec_loss = criterion(output, spec, length_mask)
        if args.n_mels > 0:
            mel_loss = criterion(output_mel, mel, length_mel_mask)
        else:
            mel_loss = 0

        train_loss = mel_loss + spec_loss

        if args.perceptual_loss > 0:
            pe_loss = perceptual_entropy(output, real, imag)
            final_loss = args.perceptual_loss * pe_loss + (1 - args.perceptual_loss) * train_loss
        else:
            final_loss = train_loss

        optimizer.zero_grad()
        final_loss.backward()

        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
        if args.optimizer == "noam":
            optimizer.step_and_update_lr()
        else:
            optimizer.step()

        losses.update(train_loss.item(), phone.size(0))
        spec_losses.update(spec_loss.item(), phone.size(0))
        if args.perceptual_loss > 0:
            pe_losses.update(pe_loss.item(), phone.size(0))
        if args.n_mels > 0:
            mel_losses.update(mel_loss.item(), phone.size(0))

        if step % args.train_step_log == 0:
            end = time.time()
            if args.model_type == "PureTransformer_norm":
                spec,_ = model.normalizer.inverse(spec,length)
                output,_= model.normalizer.inverse(output,length)
            else:
                pass
            log_figure(step, output, spec, att, length, log_save_dir, args)
            out_log = "step {}: train_loss {}; spec_loss {}; ".format(step,
                                                                      losses.avg, spec_losses.avg)
            if args.perceptual_loss > 0:
                out_log += "pe_loss {}; ".format(pe_losses.avg)
            if args.n_mels > 0:
                out_log += "mel_loss {}; ".format(mel_losses.avg)
            print("{} -- sum_time: {}s".format(out_log, (end-start)))

    info = {'loss': losses.avg, 'spec_loss': spec_losses.avg}
    if args.perceptual_loss > 0:
        info['pe_loss'] = pe_losses.avg
    if args.n_mels > 0:
        info['mel_loss'] = mel_losses.avg
    return info


def validate(dev_loader, model, device, criterion, perceptual_entropy, epoch, args):
    losses = AverageMeter()
    spec_losses = AverageMeter()
    if args.perceptual_loss > 0:
        pe_losses = AverageMeter()
    if args.n_mels > 0:
        mel_losses = AverageMeter()
    model.eval()

    log_save_dir = os.path.join(args.model_save_dir, "epoch{}/log_val_figure".format(epoch))
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    start = time.time()

    with torch.no_grad():
        for step, (phone, beat, pitch, spec, real, imag, length, chars, char_len_list, mel) in enumerate(dev_loader, 1):
            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            spec = spec.to(device).float()
            mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = length.unsqueeze(2)
            length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            length_mel_mask = length_mel_mask.to(device)
            length = length.to(device)
            char_len_list = char_len_list.to(device)
    
            if not args.use_asr_post:
                chars = chars.to(device)
                char_len_list = char_len_list.to(device)
            else:
                phone = phone.float()
            
            if args.model_type == "GLU_Transformer":
                output, att, output_mel = model(chars, phone, pitch, beat, pos_char=char_len_list,
                           pos_spec=length)
            elif args.model_type == "LSTM":
                output, hidden, output_mel = model(phone, pitch, beat)
                att = None
            elif args.model_type == "PureTransformer":
                output, att, output_mel = model(chars, phone, pitch, beat, pos_char=char_len_list,
                           pos_spec=length)

            elif args.model_type == "PureTransformer_norm":
                output,att,output_mel,spec_norm,mel_norm = model(spec,mel,chars,phone,pitch,beat,pos_char=char_len_list,pos_spec=length)
                output,_ = model.normalizer.inverse(output,length)
                #output_mel,_ = model.mel_normalizer.inverse(output_mel)
                # FIX ME, add mel.normalize

            spec_loss = criterion(output, spec, length_mask)
            if args.n_mels > 0:
                mel_loss = criterion(output_mel, mel, length_mel_mask)
            else:
                mel_loss = 0
    
            dev_loss = mel_loss + spec_loss
    
            if args.perceptual_loss > 0:
                pe_loss = perceptual_entropy(output, real, imag)
                final_loss = args.perceptual_loss * pe_loss + (1 - args.perceptual_loss) * dev_loss
            else:
                final_loss = dev_loss

            losses.update(dev_loss.item(), phone.size(0))
            spec_losses.update(spec_loss.item(), phone.size(0))
            if args.perceptual_loss > 0:
                pe_loss = perceptual_entropy(output, real, imag)
                pe_losses.update(pe_loss.item(), phone.size(0))
            if args.n_mels > 0:
                mel_losses.update(mel_loss.item(), phone.size(0))

            if step % args.dev_step_log == 0:
                log_figure(step, output, spec, att, length, log_save_dir, args)
                out_log = "step {}: train_loss {}; spec_loss {}; ".format(step, losses.avg,
                                                                          spec_losses.avg)
                if args.perceptual_loss > 0:
                    out_log += "pe_loss {}; ".format(pe_losses.avg)
                if args.n_mels > 0:
                    out_log += "mel_loss {}; ".format(mel_losses.avg)
                end = time.time()
                print("{} -- sum_time: {}s".format(out_log, (end-start)))

    info = {'loss': losses.avg, 'spec_loss': spec_losses.avg}
    if args.perceptual_loss > 0:
        info['pe_loss'] = pe_losses.avg
    if args.n_mels > 0:
        info['mel_loss'] = mel_losses.avg
    return info


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, model_filename):
    torch.save(state, model_filename)
    return 0


def record_info(train_info, dev_info, epoch, logger):
    loss_info = {
        "train_loss": train_info['loss'],
        "dev_loss": dev_info['loss']}
    logger.add_scalars("losses", loss_info, epoch)
    return 0


def invert_spectrogram(spectrogram, win_length, hop_length):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def griffin_lim(spectrogram, iter_vocoder, n_fft, hop_length, win_length):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(iter_vocoder):
        X_t = invert_spectrogram(X_best, win_length, hop_length)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, win_length, hop_length)
    y = np.real(X_t)
    return y


def spectrogram2wav(mag, max_db, ref_db, preemphasis, power, sr, hop_length, win_length, n_fft):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    hop_length = int(hop_length * sr)
    win_length = int(win_length * sr)
    n_fft = n_fft

    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag** power, 100, n_fft, hop_length, win_length)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def log_figure(step, output, spec, att, length, save_dir, args):
    # only get one sample from a batch
    # save wav and plot spectrogram
    output = output.cpu().detach().numpy()[0]
    out_spec = spec.cpu().detach().numpy()[0]
    length = np.max(length.cpu().detach().numpy()[0])
    output = output[:length]
    out_spec = out_spec[:length]
    wav = spectrogram2wav(output, args.max_db, args.ref_db, args.preemphasis, args.power, args.sampling_rate, args.frame_shift, args.frame_length, args.nfft)
    wav_true = spectrogram2wav(out_spec, args.max_db, args.ref_db, args.preemphasis, args.power, args.sampling_rate, args.frame_shift, args.frame_length, args.nfft)
    write_wav(os.path.join(save_dir, '{}.wav'.format(step)), wav, args.sampling_rate)
    write_wav(os.path.join(save_dir, '{}_true.wav'.format(step)), wav_true, args.sampling_rate)
    plt.subplot(1, 2, 1)
    specshow(output.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(out_spec.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, '{}.png'.format(step)))
    if att is not None:
        att = att.cpu().detach().numpy()[0]
        att = att[:, :length, :length]
        plt.subplot(1, 4, 1)
        specshow(att[0])
        plt.subplot(1, 4, 2)
        specshow(att[1])
        plt.subplot(1, 4, 3)
        specshow(att[2])
        plt.subplot(1, 4, 4)
        specshow(att[3])
        plt.savefig(os.path.join(save_dir, '{}_att.png'.format(step)))
