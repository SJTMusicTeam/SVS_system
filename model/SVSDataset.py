#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi, Shuai Guo)

from torch.utils.data import Dataset
import numpy as np
import torch
import os
import librosa


def _get_spectrograms(fpath, require_sr, preemphasis, n_fft, hop_length, win_length, max_db, ref_db):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath)
    if sr != require_sr:
        y = librosa.resample(y, sr, require_sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # to decibel
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mag = mag.astype(np.float32)  # (T, 1+n_fft//2)

    return mag


def _phone2char(phone_list):
    char_list = []
    for phones in phone_list:
        ini = -1
        chars = []
        for phone in phones:
            if phone != ini:
                chars.append(phone)
                ini = phone
        char_list.append(chars)
    return char_list



class SVSCollator(object):
    def __init__(self, max_len, char_max_len=50):
        self.max_len = max_len
        self.char_max_len = char_max_len

    def __call__(self, batch):
        batch_size = len(batch)
        # get spectrum dim
        spec_dim = len(batch[0][3][0])
        len_list = [len(batch[i][0]) for i in range(batch_size)]
        char_len_list = [len(batch[i][4]) for i in range(batch_size)]
        spec = np.zeros((batch_size, self.max_len, spec_dim))
        phone = np.zeros((batch_size, self.max_len))
        pitch = np.zeros((batch_size, self.max_len))
        beat = np.zeros((batch_size, self.max_len))
        chars = np.zeros((batch_size, self.char_max_len))
        for i in range(batch_size):
            length = len_list[i]
            spec[i, :length, :] = batch[i][3]
            pitch[i, :length] = batch[i][2]
            beat[i, :length] = batch[i][1]
            phone[i, :length] = batch[i][0]
            chars[i, :len(batch[i][4])] = batch[i][4]

        length = np.array(len_list)
        char_len_list = np.array(char_len_list)
        spec = torch.from_numpy(spec)
        length = torch.from_numpy(length)
        pitch = torch.from_numpy(pitch).unsqueeze(dim=-1)
        beat = torch.from_numpy(beat).unsqueeze(dim=-1)
        phone = torch.from_numpy(phone).unsqueeze(dim=-1)
        chars = torch.from_numpy(chars).unsqueeze(dim=-1)
        char_len_list = torch.from_numpy(char_len_list)
        return phone, beat, pitch, spec, length, chars, char_len_list


class SVSDataset(Dataset):
    def __init__(self,
                 align_root_path,
                 pitch_beat_root_path,
                 wav_root_path,
                 sr = 44100,
                 preemphasis = 0.97,
                 frame_shift = 0.03,
                 frame_length = 0.06,
                 n_mels = 80,
                 power = 1.2,
                 max_db = 100,
                 ref_db = 20):
        self.align_root_path = align_root_path
        self.pitch_beat_root_path = pitch_beat_root_path
        self.wav_root_path = wav_root_path
        self.sr = sr
        self.preemphasis = preemphasis
        self.frame_shift = int(frame_shift * sr)
        self.frame_length = int(frame_length * sr)
        self.n_mels = n_mels
        self.power = power
        self.max_db = max_db
        self.ref_db = ref_db

        # TODO: sum up the data source to one directory
        # get file_list
        filename_list = os.listdir(align_root_path)
        # fast debug
        filename_list = filename_list[:10]
        path_list = []
        phone_list, beat_list, pitch_list, spectrogram_list = [], [], [], []

        for i in range(len(filename_list)):
            # TODO: reload data for each get_item to handle large data
            # TODO： pre-compute the feature
            if filename_list[i][-1] != 'm' and filename_list[i][-1] != 'e':
                path = os.path.join(align_root_path, filename_list[i])
                path_list.append(path)

                with open(path, 'r') as f:
                    phone = f.read().strip().split(" ")
                    f.close()
                beat_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4],
                                         filename_list[i][4:] + "_beats.txt")
                with open(beat_path, 'r') as f:
                    beat = f.read().strip().split(" ")
                    f.close()
                pitch_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4],
                                          filename_list[i][4:] + "_pitches.txt")
                with open(pitch_path, 'r') as f:
                    pitch = f.read().strip().split(" ")
                    f.close()
                wav_path = os.path.join(wav_root_path, filename_list[i][1:4], filename_list[i][4:] + ".wav")
                spectrogram = _get_spectrograms(wav_path, self.sr, self.preemphasis,
                                                self.frame_length, self.frame_shift, self.frame_length,
                                                self.max_db, self.ref_db)

                # length check
                if np.abs(len(phone) - np.shape(spectrogram)[0]) > 3:
                    print("error file: %s" %filename_list[i])
                    continue
                # assert np.abs(len(phone) - np.shape(spectrogram)[0]) < 5
                min_length = min(len(phone), np.shape(spectrogram)[0])
                phone_list.append(phone[:min_length])
                beat_list.append(beat[:min_length])
                pitch_list.append(pitch[:min_length])
                spectrogram_list.append(spectrogram[:min_length, :])

        # sort by length desc
        length = []
        for i in range(len(phone_list)):
            length.append(len(phone_list[i]))

        self.phone_list = [x for _, x in sorted(zip(length, phone_list), reverse=True)]
        self.beat_list = [x for _, x in sorted(zip(length, beat_list), reverse=True)]
        self.pitch_list = [x for _, x in sorted(zip(length, pitch_list), reverse=True)]
        self.spectrogram_list = [x for _, x in sorted(zip(length, spectrogram_list), key=lambda x: x[0], reverse=True)]
        self.char_list = _phone2char(self.phone_list)

    def __len__(self):
        return len(self.phone_list)

    def __getitem__(self, i):
        return self.phone_list[i], self.beat_list[i], self.pitch_list[i], self.spectrogram_list[i], self.char_list[i]


