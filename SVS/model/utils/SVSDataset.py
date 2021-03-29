"""Copyright [2020] [Jiatong Shi & Shuai Guo].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# !/usr/bin/env python3

import librosa
import logging
from math import log2
from math import pow
import numpy as np
import os
import random
from SVS.model.utils.utils import melspectrogram
import torch
from torch.utils.data import Dataset


def _get_spectrograms(
    fpath,
    require_sr,
    preemphasis,
    n_fft,
    hop_length,
    win_length,
    max_db,
    ref_db,
    n_mels=80,
):
    """Parse the wave file in `fpath` and.

    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    """
    # Loading sound file
    y, sr = librosa.load(fpath, sr=None)
    if sr != require_sr:
        y = librosa.resample(y, sr, require_sr)

    if n_mels > 0:
        # mel_basis = librosa.filters.mel(require_sr, n_fft, n_mels)
        # mel = np.dot(mel_basis, mag)  # (n_mels, t)
        # mel = 20 * np.log10(np.maximum(1e-5, mel))
        # mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
        # mel = mel.T.astype(np.float32)
        mel = melspectrogram(y, n_fft, hop_length, win_length, require_sr, n_mels)
        mel = mel.T.astype(np.float32)

    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

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

    if n_mels > 0:
        return mag, mel, phase
    else:
        return mag, None, phase


def _load_sing_quality(quality_file, standard=3):
    """_load_sing_quality."""
    quality = []
    with open(quality_file, "r") as f:
        data = f.read().split("\n")[1:]
        data = list(map(lambda x: x.split(","), data))
        for sample in data:
            if sample[1] != "" and int(sample[1]) >= standard:
                quality.append("0" * (4 - len(sample[0])) + sample[0])
    return quality


def _phone2char(phones, char_max_len):
    """_phone2char."""
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


def _Hz2Semitone(freq):
    """_Hz2Semitone."""
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    if freq == 0:
        return "Sil"  # silence
    else:
        h = round(12 * log2(freq / C0))
        octave = h // 12
        n = h % 12
        return name[n] + "_" + str(octave)


def _full_semitone_list(semitone_min, semitone_max):
    """_full_semitone_list."""
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    name_min, octave_min = semitone_min.split("_")
    name_max, octave_max = semitone_max.split("_")

    assert octave_min <= octave_max

    res = ["Sil"]
    flag_insert = 0
    for octave in range(int(octave_min), int(octave_max) + 1):
        for res_name in name:
            if res_name == name_min and octave == int(octave_min):
                flag_insert = 1
            elif res_name == name_max and octave == int(octave_max):
                res.append(res_name + "_" + str(octave))
                flag_insert = 0
                break
            if flag_insert == 1:
                res.append(res_name + "_" + str(octave))

    return res


def _calculate_phone_element_freq(phone_array):
    """Return the phone list and freq of given phone_array."""
    phone_list = [
        phone_array[index]
        for index in range(len(phone_array))
        if index == 0 or phone_array[index] != phone_array[index - 1]
    ]
    phone_freq = []

    begin_index = 0
    for phone in phone_list:
        freq = 0
        for index in range(begin_index, len(phone_array)):
            if phone_array[index] == phone:
                freq += 1
            else:
                phone_freq.append(freq)
                begin_index = index
                break
    phone_freq.append(freq)

    assert len(phone_list) == len(phone_freq)

    return phone_list, phone_freq


def _phone_shift(phone_array, phone_shift_size):

    phone_list, phone_freq = _calculate_phone_element_freq(phone_array)

    shift_side = random.randint(0, 1)  # 0 - left, 1 - right
    # do phone shift augment
    for index in range(len(phone_freq)):

        shift_size = random.randint(1, phone_shift_size)
        flag_shift = 1 if random.random() > 0.5 else 0

        if flag_shift:
            if phone_freq[index] > 2 * shift_size:

                if shift_side == 0 and index != 0:
                    # left shift
                    phone_freq[index] -= shift_size
                    phone_freq[index - 1] += shift_size
                elif shift_side == 1 and index != len(phone_freq) - 1:
                    # right shift
                    phone_freq[index] -= shift_size
                    phone_freq[index + 1] += shift_size

    # reconstruct phone array based on its freq
    res = []
    for index in range(len(phone_freq)):
        for freq in range(phone_freq[index]):
            res.append(phone_list[index])
    res = np.array(res)

    assert len(res) == len(phone_array)

    return res


def _pitch_shift(f0_array, semitone_list):

    f0_list = [f0 for f0 in f0_array if f0 != 0]

    if len(f0_list) == 0:
        # no shift
        return [semitone_list.index(_Hz2Semitone(f0)) for f0 in f0_array]

    f0_min = np.min(f0_list)
    f0_max = np.max(f0_list)

    semitone_min = _Hz2Semitone(f0_min)
    semitone_max = _Hz2Semitone(f0_max)

    index_min = semitone_list.index(semitone_min)
    index_max = semitone_list.index(semitone_max)

    flag_left, flag_right = False, False
    if index_min - 12 >= 1:
        flag_left = True
    if index_max + 12 <= len(semitone_list) - 1:
        flag_right = True

    # decide shift direction
    if flag_left is True and flag_right is True:
        shift_side = random.randint(0, 1)  # 0 - left, 1 - right
    elif flag_left is True:
        shift_side = 0
    elif flag_right is True:
        shift_side = 1
    else:
        shift_side = -1

    # decide whether to shift
    flag_shift = 1 if random.random() > 0.5 else 0

    if shift_side == -1 or flag_shift == 0:
        # no shift
        return [semitone_list.index(_Hz2Semitone(f0)) for f0 in f0_array]
    else:
        if shift_side == 0:
            # left shift
            res = []
            for f0 in f0_array:
                if f0 == 0:
                    res.append(semitone_list.index(_Hz2Semitone(f0)))
                else:
                    res.append(semitone_list.index(_Hz2Semitone(f0)) - 12)
            return res
        elif shift_side == 1:
            # right shift
            res = []
            for f0 in f0_array:
                if f0 == 0:
                    res.append(semitone_list.index(_Hz2Semitone(f0)))
                else:
                    res.append(semitone_list.index(_Hz2Semitone(f0)) + 12)
            return res


class SVSCollator(object):
    """SVSCollator."""

    def __init__(
        self,
        max_len,
        char_max_len=80,
        use_asr_post=False,
        phone_size=68,
        n_mels=80,
        db_joint=False,
        random_crop=False,
        crop_min_length=100,
        Hz2semitone=False,
    ):
        """init."""
        self.max_len = max_len
        # plus 1 for aligner to consider padding char
        self.char_max_len = char_max_len + 1
        self.use_asr_post = use_asr_post
        self.phone_size = phone_size - 1
        self.n_mels = n_mels
        self.db_joint = db_joint
        self.random_crop = random_crop
        self.crop_min_length = crop_min_length
        self.Hz2semitone = Hz2semitone

        assert crop_min_length <= max_len

    def __call__(self, batch):
        """call."""
        # phone, beat, pitch, spectrogram, char, phase, mel

        batch_size = len(batch)
        # get spectrum dim
        spec_dim = len(batch[0]["spec"][0])
        len_list = [len(batch[i]["phone"]) for i in range(batch_size)]
        spec = np.zeros((batch_size, self.max_len, spec_dim))
        real = np.zeros((batch_size, self.max_len, spec_dim))
        imag = np.zeros((batch_size, self.max_len, spec_dim))
        if self.n_mels > 0:
            mel = np.zeros((batch_size, self.max_len, self.n_mels))
        pitch = np.zeros((batch_size, self.max_len))
        beat = np.zeros((batch_size, self.max_len))
        length_mask = np.zeros((batch_size, self.max_len))
        semitone = np.zeros((batch_size, self.max_len))
        
        filename_list = [batch[i]["filename"] for i in range(batch_size)]

        if self.db_joint:
            singer_id = [batch[i]["singer_id"] for i in range(batch_size)]

        if self.use_asr_post:
            phone = np.zeros((batch_size, self.max_len, self.phone_size))
        else:
            # char_len_list=[len(batch[i]["char"]) for i in range(batch_size)]
            phone = np.zeros((batch_size, self.max_len))
            chars = np.zeros((batch_size, self.char_max_len))
            char_len_mask = np.zeros((batch_size, self.char_max_len))

        for i in range(batch_size):
            crop_length = random.randint(self.crop_min_length, self.max_len)

            if self.random_crop and crop_length < len_list[i]:
                # want2cut length < G.T. length
                index_begin = random.randint(0, int(len_list[i] - crop_length))
                index_end = index_begin + crop_length

                length_mask[i, :crop_length] = np.arange(1, crop_length + 1)
                spec[i, :crop_length, :] = batch[i]["spec"][
                    index_begin:index_end
                ]  # [begin, end)
                real[i, :crop_length, :] = batch[i]["phase"][index_begin:index_end].real
                imag[i, :crop_length, :] = batch[i]["phase"][index_begin:index_end].imag
                pitch[i, :crop_length] = batch[i]["pitch"][index_begin:index_end]
                beat[i, :crop_length] = batch[i]["beat"][index_begin:index_end]

                if self.Hz2semitone:
                    semitone[i, :crop_length] = batch[i]["semitone"][
                        index_begin:index_end
                    ]

                if self.n_mels > 0:
                    mel[i, :crop_length, :] = batch[i]["mel"][index_begin:index_end]

                if self.use_asr_post:
                    phone[i, :crop_length, :] = batch[i]["phone"][index_begin:index_end]
                else:
                    char_leng = min(len(batch[i]["char"]), self.char_max_len)
                    phone[i, :crop_length] = batch[i]["phone"][index_begin:index_end]
                    chars[i, :char_leng] = batch[i]["char"][:char_leng]
                    char_len_mask[i, :char_leng] = np.arange(1, char_leng + 1)

            else:
                length = min(len_list[i], self.max_len)

                length_mask[i, :length] = np.arange(1, length + 1)
                spec[i, :length, :] = batch[i]["spec"][:length]
                real[i, :length, :] = batch[i]["phase"][:length].real
                imag[i, :length, :] = batch[i]["phase"][:length].imag
                pitch[i, :length] = batch[i]["pitch"][:length]
                beat[i, :length] = batch[i]["beat"][:length]

                if self.Hz2semitone:
                    semitone[i, :length] = batch[i]["semitone"][:length]

                if self.n_mels > 0:
                    mel[i, :length] = batch[i]["mel"][:length]

                if self.use_asr_post:
                    phone[i, :length, :] = batch[i]["phone"][:length]
                else:
                    char_leng = min(len(batch[i]["char"]), self.char_max_len)
                    phone[i, :length] = batch[i]["phone"][:length]
                    chars[i, :char_leng] = batch[i]["char"][:char_leng]
                    char_len_mask[i, :char_leng] = np.arange(1, char_leng + 1)

        spec = torch.from_numpy(spec)
        if self.n_mels > 0:
            mel = np.array(mel).astype(np.float64)
            mel = torch.from_numpy(mel)
        else:
            mel = None
        imag = torch.from_numpy(imag)
        real = torch.from_numpy(real)
        length_mask = torch.from_numpy(length_mask).long()
        pitch = torch.from_numpy(pitch).unsqueeze(dim=-1).long()
        beat = torch.from_numpy(beat).unsqueeze(dim=-1).long()
        phone = torch.from_numpy(phone).unsqueeze(dim=-1).long()

        if not self.use_asr_post:
            chars = torch.from_numpy(chars).unsqueeze(dim=-1).to(torch.int64)
            char_len_mask = torch.from_numpy(char_len_mask).long()
        else:
            chars = None
            char_len_mask = None

        if self.Hz2semitone:
            semitone = torch.from_numpy(semitone).unsqueeze(dim=-1).long()
        else:
            semitone = None

        if self.db_joint:
            return (
                phone,
                beat,
                pitch,
                spec,
                real,
                imag,
                length_mask,
                chars,
                char_len_mask,
                mel,
                singer_id,
                semitone,
                filename_list,
            )
        else:
            return (
                phone,
                beat,
                pitch,
                spec,
                real,
                imag,
                length_mask,
                chars,
                char_len_mask,
                mel,
                semitone,
                filename_list,
            )


class SVSDataset(Dataset):
    """SVSDataset."""

    def __init__(
        self,
        align_root_path,
        pitch_beat_root_path,
        wav_root_path,
        char_max_len=80,
        max_len=500,
        sr=44100,
        preemphasis=0.97,
        nfft=2048,
        frame_shift=0.03,
        frame_length=0.06,
        n_mels=80,
        power=1.2,
        max_db=100,
        ref_db=20,
        sing_quality="conf/sing_quality.csv",
        standard=3,
        db_joint=False,
        Hz2semitone=False,
        semitone_min="F_1",
        semitone_max="D_6",
        phone_shift_size=-1,
        semitone_shift=False,
    ):
        """init."""
        self.align_root_path = align_root_path
        self.pitch_beat_root_path = pitch_beat_root_path
        self.wav_root_path = wav_root_path
        self.char_max_len = char_max_len
        self.max_len = max_len
        self.sr = sr
        self.preemphasis = preemphasis
        self.nfft = nfft
        self.frame_shift = int(frame_shift * sr)
        self.frame_length = int(frame_length * sr)
        self.n_mels = n_mels
        self.power = power
        self.max_db = max_db
        self.ref_db = ref_db
        self.db_joint = db_joint
        self.Hz2semitone = Hz2semitone
        self.phone_shift_size = phone_shift_size
        self.semitone_shift = semitone_shift

        if Hz2semitone:
            self.semitone_list = _full_semitone_list(semitone_min, semitone_max)

        if standard > 0:
            print(standard)
            quality = _load_sing_quality(sing_quality, standard)
        else:
            quality = None
        # get file_list
        self.filename_list = os.listdir(align_root_path)
        # phone_list, beat_list, pitch_list, spectrogram_list = [], [], [], []
        for filename in self.filename_list:
            if quality is None:
                break
            if filename[-4:] != ".npy" or filename[:4] not in quality:
                print("remove file {}".format(filename))
                self.filename_list.remove(filename)

    def __len__(self):
        """len."""
        return len(self.filename_list)

    def __getitem__(self, i):
        """getitem."""
        path = os.path.join(self.align_root_path, self.filename_list[i])
        try:
            phone = np.load(path)
            # phone shift augment
            if self.phone_shift_size != -1:
                assert self.phone_shift_size == 1 or self.phone_shift_size == 2
                phone = _phone_shift(phone, self.phone_shift_size)
        except Exception:
            print("error path {}".format(path))

        if self.db_joint:
            db_name = self.filename_list[i].split("_")[0]
            if db_name == "hts":
                singer_id = 0
            elif db_name == "jsut":
                singer_id = 1
            elif db_name == "kiritan":
                singer_id = 2
            elif db_name == "natsume":
                singer_id = 3
            elif db_name == "pjs":
                singer_id = 4
            elif db_name == "ofuton":
                singer_id = 5
            elif db_name == "oniku":
                singer_id = 6
            else:
                raise ValueError(
                    "ValueError exception thrown, No such dataset: ", db_name
                )

            beat_path = os.path.join(
                self.pitch_beat_root_path, self.filename_list[i][:-4] + "_beats.npy"
            )
            beat_numpy = np.load(beat_path)
            beat_index = list(map(lambda x: int(x), beat_numpy))
            beat = np.zeros(len(phone))
            beat[beat_index] = 1
            pitch_path = os.path.join(
                self.pitch_beat_root_path, self.filename_list[i][:-4] + "_pitch.npy"
            )
            pitch = np.load(pitch_path)
            wav_path = os.path.join(
                self.wav_root_path, self.filename_list[i][:-4] + ".wav"
            )
        else:
            # path is different between combine-db <-> single db
            beat_path = os.path.join(
                self.pitch_beat_root_path,
                str(int(self.filename_list[i][1:4])),
                self.filename_list[i][4:-4] + "_beats.npy",
            )
            beat_numpy = np.load(beat_path)
            beat_index = list(map(lambda x: int(x), beat_numpy))
            beat = np.zeros(len(phone))
            beat[beat_index] = 1
            pitch_path = os.path.join(
                self.pitch_beat_root_path,
                str(int(self.filename_list[i][1:4])),
                self.filename_list[i][4:-4] + "_pitch.npy",
            )
            pitch = np.load(pitch_path)
            wav_path = os.path.join(
                self.wav_root_path,
                str(int(self.filename_list[i][1:4])),
                self.filename_list[i][4:-4] + ".wav",
            )

        spectrogram, mel, phase = _get_spectrograms(
            wav_path,
            self.sr,
            self.preemphasis,
            self.nfft,
            self.frame_shift,
            self.frame_length,
            self.max_db,
            self.ref_db,
            n_mels=self.n_mels,
        )

        # length check
        if np.abs(len(phone) - np.shape(spectrogram)[0]) > 3:
            logging.info("error file: %s" % self.filename_list[i])
            logging.info(
                "spectrum_size: {}, alignment_size: {}, "
                "pitch_size: {}, beat_size: {}".format(
                    np.shape(spectrogram)[0], len(phone), len(pitch), len(beat)
                )
            )
        # fix me
        # assert np.abs(len(phone) - np.shape(spectrogram)[0]) <= 15
        # for post condition
        if len(phone.shape) > 1:
            char, trimed_length = None, len(phone)
        else:
            char, trimed_length = _phone2char(phone[: self.max_len], self.char_max_len)
        min_length = min(
            len(phone), np.shape(spectrogram)[0], trimed_length, len(pitch), len(beat)
        )
        phone = phone[:min_length]
        beat = beat[:min_length]
        pitch = pitch[:min_length]
        spectrogram = spectrogram[:min_length, :]
        phase = phase[:min_length, :]

        if mel is not None:
            mel = mel[:min_length, :]

        if self.Hz2semitone:
            semitone = [self.semitone_list.index(_Hz2Semitone(f0)) for f0 in pitch]
            if self.semitone_shift:
                semitone = _pitch_shift(pitch, self.semitone_list)
        else:
            semitone = None

        # print("char len: {}, phone len: {}, spectrom: {}"
        # .format(len(char), len(phone), np.shape(spectrogram)[0]))
        # logging.info(min_length)

        if self.db_joint:
            return {
                "phone": phone,
                "beat": beat,
                "pitch": pitch,
                "spec": spectrogram,
                "char": char,
                "phase": phase,
                "mel": mel,
                "singer_id": singer_id,
                "semitone": semitone,
                "filename": self.filename_list[i][:-4]
            }
        else:
            return {
                "phone": phone,
                "beat": beat,
                "pitch": pitch,
                "spec": spectrogram,
                "char": char,
                "phase": phase,
                "mel": mel,
                "semitone": semitone,
                "filename": self.filename_list[i][:-4]
            }
