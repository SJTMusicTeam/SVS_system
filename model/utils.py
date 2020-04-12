#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import torch
import numpy as np
import copy
import time
import librosa
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_src_key_padding_mask(src_len, max_len):
    bs = len(src_len)
    mask = np.zeros((bs, max_len))
    for i in range(bs):
        mask[i, src_len[i]:] = 1
    return torch.from_numpy(mask).float()


def train_one_epoch(train_loader, model, device, optimizer, criterion, args):
    losses = AverageMeter()
    model.train()
    start = time.time()
    for step, (phone, beat, pitch, spec, length, chars, char_len_list) in enumerate(train_loader, 1):
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

        output = model(chars, phone, pitch, beat, src_key_padding_mask=length,
                       char_key_padding_mask=char_len_list)

        train_loss = criterion(output, spec, length_mask)

        optimizer.zero_grad()
        train_loss.backward()
        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
        optimizer.step_and_update_lr()
        losses.update(train_loss.item(), phone.size(0))
        if step % 100 == 0:
            end = time.time()
            print("step {}: train_loss {} -- sum_time: {}s".format(step, losses.avg, end - start))

    info = {'loss': losses.avg}
    return info


def validate(dev_loader, model, device, criterion, args):
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (phone, beat, pitch, spec, length, chars, char_len_list) in enumerate(dev_loader, 1):
            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            spec = spec.to(device).float()
            chars = chars.to(device)
            length = length.to(device)
            length_mask = create_src_key_padding_mask(length, args.num_frames)
            length_mask = length_mask.unsqueeze(2)
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            char_len_list = char_len_list.to(device).float()

            output = model(chars, phone, pitch, beat, src_key_padding_mask=length,
                           char_key_padding_mask=char_len_list)

            train_loss = criterion(output, spec, length_mask)
            losses.update(train_loss.item(), phone.size(0))
            if step % 10 == 0:
                print("step {}: {}".format(step, losses.avg))

    info = {'loss': losses.avg}
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


def spectrogram2wav(mag, max_db, ref_db, preemphasis, power):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag** power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    # wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)
