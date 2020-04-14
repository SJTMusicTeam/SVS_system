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
    y, sr = librosa.load(fpath, sr=None)
    if sr != require_sr:
        y = librosa.resample(y, sr, require_sr)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag, phase = librosa.magphase(linear)
    # mag = np.abs(linear)  # (1+n_fft//2, T)

    # to decibel
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    phase = phase.T

    return mag, phase


def _phone2char(phones, char_max_len):
    ini = -1
    chars = []
    phones_index = 0
    for phone in phones:
        if phone != ini:
            chars.append(phone)
            ini = phone
        phones_index += 1
        if len(chars) == char_max_len:
            break
    return chars, phones_index



class SVSCollator(object):
    def __init__(self, max_len, char_max_len=80):
        self.max_len = max_len
        # plus 1 for aligner to consider padding char
        self.char_max_len = char_max_len + 1

    def __call__(self, batch):
        batch_size = len(batch)
        # get spectrum dim
        spec_dim = len(batch[0][3][0])
        len_list = [len(batch[i][0]) for i in range(batch_size)]
        char_len_list = [len(batch[i][4]) for i in range(batch_size)]
        spec = np.zeros((batch_size, self.max_len, spec_dim))
        phase = np.zeros((batch_size, self.max_len, spec_dim))
        phone = np.zeros((batch_size, self.max_len))
        pitch = np.zeros((batch_size, self.max_len))
        beat = np.zeros((batch_size, self.max_len))
        chars = np.zeros((batch_size, self.char_max_len))
        for i in range(batch_size):
            length = min(len_list[i], self.max_len)
            char_leng = min(len(batch[i][4]), self.char_max_len)
            spec[i, :length, :] = batch[i][3][:length]
            phase[i, :length, :] = batch[i][5][:length]
            pitch[i, :length] = batch[i][2][:length]
            beat[i, :length] = batch[i][1][:length]
            phone[i, :length] = batch[i][0][:length]
            chars[i, :char_leng] = batch[i][4][:char_leng]

        length = np.array(len_list)
        char_len_list = np.array(char_len_list)
        spec = torch.from_numpy(spec)
        imag = torch.from_numpy(phase.imag)
        real = torch.from_numpy(phase.real)
        length = torch.from_numpy(length)
        pitch = torch.from_numpy(pitch).unsqueeze(dim=-1).long()
        beat = torch.from_numpy(beat).unsqueeze(dim=-1).long()
        phone = torch.from_numpy(phone).unsqueeze(dim=-1).long()
        chars = torch.from_numpy(chars).unsqueeze(dim=-1).to(torch.int64)
        char_len_list = torch.from_numpy(char_len_list)
        return phone, beat, pitch, spec, real, imag, length, chars, char_len_list


class SVSDataset(Dataset):
    def __init__(self,
                 align_root_path,
                 pitch_beat_root_path,
                 wav_root_path,
                 char_max_len = 80,
                 max_len = 500,
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
        self.char_max_len = char_max_len
        self.max_len = max_len
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
        self.filename_list = os.listdir(align_root_path)
        # fast debug
        # self.filename_list = self.filename_list[:10]
        phone_list, beat_list, pitch_list, spectrogram_list = [], [], [], []
        for filename in self.filename_list:
            if filename[-1] == 'm' or filename[-1] == 'e':
                self.filename_list.remove(filename)


    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, i):
        path = os.path.join(self.align_root_path, self.filename_list[i])
        phone = np.load(path)
        beat_path = os.path.join(self.pitch_beat_root_path, str(int(self.filename_list[i][1:4])),
                                 self.filename_list[i][4:-4] + "_beats.npy")
        beat_numpy = np.load(beat_path)
        beat_index = list(map(lambda x : int(x), beat_numpy))
        beat = np.zeros(len(phone))
        beat[beat_index] = 1
        pitch_path = os.path.join(self.pitch_beat_root_path, str(int(self.filename_list[i][1:4])),
                                  self.filename_list[i][4:-4] + "_pitch.npy")
        pitch = np.load(pitch_path)
        wav_path = os.path.join(self.wav_root_path, str(int(self.filename_list[i][1:4])), self.filename_list[i][4:-4] + ".wav")

        spectrogram, phase = _get_spectrograms(wav_path, self.sr, self.preemphasis,
                                        self.frame_length, self.frame_shift, self.frame_length,
                                        self.max_db, self.ref_db)

        # length check
        
        if np.abs(len(phone) - np.shape(spectrogram)[0]) > 3:
            print("error file: %s" %self.filename_list[i])
            print("spectrum_size: {}, alignment_size: {}, pitch_size: {}, beat_size: {}".format(np.shape(spectrogram)[0],
                  len(phone), len(pitch), len(beat)))
        assert np.abs(len(phone) - np.shape(spectrogram)[0]) < 5
        char, trimed_length = _phone2char(phone[:self.max_len], self.char_max_len)
        min_length = min(len(phone), np.shape(spectrogram)[0], trimed_length)
        phone = phone[:min_length]
        beat = beat[:min_length]
        pitch = pitch[:min_length]
        spectrogram = spectrogram[:min_length, :]
        phase = phase[:min_length, :]
        # print("char len: {}, phone len: {}, spectrom: {}".format(len(char), len(phone), np.shape(spectrogram)[0]))

        return phone, beat, pitch, spectrogram, char, phase

