#!/usr/bin/env python
# coding: utf-8

import os
import librosa
import numpy as np


def _get_spectrograms(
    fpath, require_sr, preemphasis, n_fft, hop_length, win_length, max_db, ref_db
):
    """Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    """
    # Loading sound file
    y, sr = librosa.load(fpath)
    if sr != require_sr:
        y = librosa.resample(y, sr, require_sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # to decibel
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mag


def save_spectrograms(
    wav_path, save_path, sr, preemphasis, frame_shift, frame_length, max_db, ref_db
):
    os.mkdir(save_path)
    for root, dirs, files in os.walk(wav_root_path):
        for d in dirs:
            os.mkdir(os.path.join(save_path, d))
        for f in files:
            if f[-4:] == ".wav":
                wav_path = os.path.join(root, f)
                # print(wav_path)
                spectrogram = _get_spectrograms(
                    wav_path,
                    sr,
                    preemphasis,
                    int(frame_length * sr),
                    int(frame_shift * sr),
                    int(frame_length * sr),
                    max_db,
                    ref_db,
                )
                save_file_path = os.path.join(save_path, os.path.split(root)[1])
                save_file_path = os.path.join(save_file_path, f[:-4] + ".npy")
                print(save_file_path)
                np.save(save_file_path, spectrogram)


"""
wav_root_path = 'wav_clean'
sr = 44100
preemphasis = 0.97
frame_shift = 0.03
frame_length = 0.06
n_mels = 80
power = 1.2
max_db = 100
ref_db = 20
"""

if __name__ == "__main__":
    wav_path = "wav_clean"
    save_path = os.path.join("spectrograms", wav_path)
    os.mkdir("spectrograms")
    save_spectrograms(
        wav_path="wav_clean",
        save_path=save_path,
        sr=44100,
        preemphasis=0.97,
        frame_shift=0.03,
        frame_length=0.06,
        max_db=100,
        ref_db=20,
    )
