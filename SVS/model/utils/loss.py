"""Copyright [2020] [Jiatong Shi].

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

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import librosa
import numpy as np
import torch
from torch import nn


class MaskedLoss(torch.nn.Module):
    """MaskedLoss."""

    def __init__(self, loss, mask_free=False):
        """init."""
        super(MaskedLoss, self).__init__()
        self.loss = loss
        self.mask_free = mask_free

    def forward(self, output, target, length):
        """forward."""
        if self.mask_free:
            if self.loss == "mse":
                return torch.mean((output - target) ** 2.0)
            elif self.loss == "l1":
                return torch.mean(torch.abs(output - target))
        else:
            output = torch.mul(output, length)
            target = torch.mul(target, length)
            if self.loss == "mse":
                return torch.sum((output - target) ** 2.0) / torch.sum(length)
            elif self.loss == "l1":
                return torch.sum(torch.abs(output - target)) / torch.sum(length)


def tq(f_bin, fs, fft_bins):
    """tq."""
    f = (np.array(f_bin) + 1) * fs / fft_bins
    # p1 = 3.64 * ((f / 1000) ** -0.8)
    # p2 = 6.5 * np.exp(-0.6 * ((f / 1000 - 3.3) ** 2))
    # p3 = (10 ** -3) * ((f / 1000) ** 4)
    # th = p1 + p2 + p3
    th = (
        3.64 * ((f / 1000 + 0.000001) ** -0.8)
        - 6.5 * np.exp(-0.6 * ((f / 1000 - 3.3) ** 2))
        + (10 ** -3) * ((f / 1000) ** 4)
    )  # threshold of hearing formula
    return th


def cband():
    """Pre-define idealized critical band filter bank.

    :return: idealized critical band filter bank
    """
    return np.array(
        [
            0,
            100,
            200,
            300,
            400,
            510,
            630,
            770,
            920,
            1080,
            1270,
            1480,
            1720,
            2000,
            2320,
            2700,
            3150,
            3700,
            4400,
            5300,
            6400,
            7700,
            9500,
            12000,
            15500,
            22050,
        ]
    )


def cal_psd2bark_dict(fs=16000, win_len=160):
    """Compute a map from bark_band to PSD component index list.

    :param fs: sampling rate (int)
    :param win_len: window length (int)
    :return: return (psd_list, bark_num) where psd_list is
        {bark_band_index: [spectrum_start, spectrum_end]} and bark
    number is the number of bark available corresponding to the sampling rate
    """
    # for current form, only less than 44100 sr can be processed
    assert fs <= 44100
    unit = fs // win_len
    bw = (cband())[1:]
    index = 1
    bark_num = 0
    psd_list = {}
    for i in range(len(bw)):
        if index >= win_len // 2:
            bark_num = i + 1
            break

        if index * unit <= bw[i]:
            start = index
        else:
            continue

        while index * unit <= bw[i] and index <= win_len // 2:
            index += 1
        end = index - 1

        psd_list[i + 1] = (start, end)

    return psd_list, bark_num


def cal_spread_function(bark_num):
    """Calculate spread function.

    :param bark_num: point number used in analysis (int)
    :return: torch.Tensor()
    """
    bark_use = bark_num - 1
    sf = np.zeros((bark_use, bark_use))
    for i in range(bark_use):
        for j in range(bark_use):
            sf[i][j] = 10 ** (
                (
                    15.81
                    + 7.5 * (i - j + 0.474)
                    - 17.5 * np.sqrt(1 + np.power(i - j + 0.474, 2))
                )
                / 10
            )
    return torch.Tensor(sf.T)


def geomean(iterable):
    """Calculate geometric mean of a given iterable.

    :param iterable: a torch.Tensor with one dimension
    :return: the geometric mean of a given iterable
    """
    # sign = torch.sign(iterable)
    temp = torch.sum(torch.log10(torch.add(torch.abs(iterable), 1e-30)), -1)
    temp = torch.mul(temp, 1 / iterable.size()[-1])
    # return torch.mul(torch.pow(10, temp), sign)
    return torch.pow(10, temp)


def arimean(iterable):
    """Calculate arithmetic mean of a given iterable.

    :param iterable: a torch.Tensor with one dimension
    :return: the arithmetic mean of a given iterable
    """
    return torch.mean(abs(iterable), -1)


class PerceptualEntropy(nn.Module):
    """PerceptualEntropy."""

    def __init__(self, bark_num, spread_function, fs, win_len, psd_dict):
        """init."""
        super(PerceptualEntropy, self).__init__()
        self.bark_num = bark_num
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device
        spread_function = spread_function.to(device)
        self.spread_function = spread_function
        self.psd_dict = psd_dict
        self.log_term = 1e-8
        self.Tq = tq(f_bin=np.array(range(win_len // 2)), fs=fs, fft_bins=win_len)
        self.renormalize = None
        # for BWE, a cutoff should be used,
        # since we only predict the high band signal
        # self.cutoff = 17
        # depend on bark band size and the widen frequency part
        # cutoff is one-based
        self.cutoff = 1

    def forward(self, log_magnitude, real, imag):
        """forward."""
        # in case for initial turbulance, may use clamp to clip extreme value
        spectrum = torch.clamp(log_magnitude, -1000, 10)
        spectrum = torch.exp(spectrum)
        loss = torch.Tensor([0.0])
        loss.requires_grad_(True)
        loss = loss.to(self.device)
        real_parts = torch.mul(real, spectrum)
        imag_parts = torch.mul(imag, spectrum)
        renormalize = {}
        # compute spread function every time ( may be refined later )
        # currently we found an error after reuse the computing graph
        if True:
            bark_scale_band_test = torch.ones(
                spectrum.shape[0], spectrum.shape[1], self.bark_num - 1
            ).to(self.device)
            c_test = torch.matmul(bark_scale_band_test, self.spread_function)
            c_test = torch.log10(torch.add(c_test, self.log_term))

            for i in range(1, self.bark_num):
                k_test = self.psd_dict[i][1] - self.psd_dict[i][0] + 1
                specific_band_test = spectrum.narrow(2, self.psd_dict[i][0] - 1, k_test)
                geo_test = geomean(specific_band_test)
                ari_test = arimean(specific_band_test)
                spectral_flatness_measure_test = torch.mul(
                    torch.log10(
                        torch.add(torch.div(geo_test, ari_test), self.log_term)
                    ),
                    10,
                )
                alpha_test = torch.min(
                    torch.div(spectral_flatness_measure_test, -60),
                    (torch.ones(spectral_flatness_measure_test.shape)).to(self.device),
                )
                offset_test = torch.add(
                    torch.mul(alpha_test, 14.5 + i),
                    torch.mul(torch.add(torch.neg(alpha_test), 1.0), 5.5),
                )
                t_test = torch.div(torch.neg(offset_test), 10)
                t_test = torch.unsqueeze(t_test, -1)
                t_test = torch.pow(10, torch.add(c_test.narrow(2, i - 1, 1), t_test))
                renormalize[i] = t_test.repeat((1, 1, k_test))

        bark_scale_band = torch.Tensor().to(self.device)
        bark_scale_band.requires_grad_(True)
        for i in range(1, self.bark_num):
            k = self.psd_dict[i][1] - self.psd_dict[i][0] + 1
            specific_band = spectrum.narrow(2, self.psd_dict[i][0] - 1, k)
            psum = torch.sum(specific_band, dim=2)
            bark_scale_band = torch.cat(
                (bark_scale_band, torch.unsqueeze(psum, -1)), dim=2
            )

        c = torch.matmul(bark_scale_band, self.spread_function)
        c = torch.log10(torch.add(c, self.log_term))

        for i in range(self.cutoff, self.bark_num):
            k = self.psd_dict[i][1] - self.psd_dict[i][0] + 1
            specific_band = spectrum.narrow(2, self.psd_dict[i][0] - 1, k)
            geo = geomean(specific_band)
            ari = arimean(specific_band)
            spectral_flatness_measure = torch.mul(
                torch.log10(torch.add(torch.div(geo, ari), self.log_term)), 10
            )

            alpha = torch.min(
                torch.div(spectral_flatness_measure, -60),
                (torch.ones(spectral_flatness_measure.shape)).to(self.device),
            )
            offset = torch.add(
                torch.mul(alpha, 14.5 + i),
                torch.mul(torch.add(torch.neg(alpha), 1.0), 5.5),
            )
            t = torch.div(torch.neg(offset), 10)
            t = torch.unsqueeze(t, -1)
            t = torch.pow(10, torch.add(c.narrow(2, i - 1, 1), t))

            specific_real_band = real_parts.narrow(2, self.psd_dict[i][0] - 1, k)
            specific_imag_band = imag_parts.narrow(2, self.psd_dict[i][0] - 1, k)
            t = t.repeat((1, 1, k))
            bound = self.Tq[self.psd_dict[i][0] - 1 : self.psd_dict[i][0] + k - 1]
            bound = bound[np.newaxis, np.newaxis, :]
            bound = torch.Tensor(bound).repeat(t.shape[0], t.shape[1], 1)
            bound = bound.to(self.device)

            t = torch.div(t, renormalize[i].narrow(1, 0, t.shape[1]))

            t = torch.max(t, bound)
            # t.retain_grad()

            # compare to original model, we remove round,
            # since it is not supported for auto-diff
            pe_real = torch.log2(
                torch.add(
                    torch.mul(
                        torch.abs(
                            torch.div(
                                specific_real_band,
                                torch.sqrt(torch.div(torch.mul(t, 6), k)),
                            )
                        ),
                        2,
                    ),
                    1,
                )
            )

            pe_imag = torch.log2(
                torch.add(
                    torch.mul(
                        torch.abs(
                            torch.div(
                                specific_imag_band,
                                torch.sqrt(torch.div(torch.mul(t, 6), k)),
                            )
                        ),
                        2,
                    ),
                    1,
                )
            )

            loss = torch.cat((loss, torch.mean(pe_real).view(1)), 0)
            loss = torch.cat((loss, torch.mean(pe_imag).view(1)), 0)

        return torch.reciprocal(torch.add(torch.sum(loss), 1))


def _test_perceptual_entropy(filename):
    """_test_perceptual_entropy."""
    fs = 16000
    win_len = 320

    signal, sr = librosa.load(filename, None)
    signal = librosa.core.stft(
        signal,
        n_fft=win_len,
        window="hann",
        win_length=win_len,
        hop_length=win_len // 2,
    )
    signal = np.transpose(signal, (1, 0))

    # get phase, log_magnitude
    magnitude, phase = librosa.core.magphase(signal.T)
    log_magnitude = np.log(magnitude + 1e-10)

    # initial perceptual entropy
    psd_dict, bark_num = cal_psd2bark_dict(fs=fs, win_len=win_len)
    sf = cal_spread_function(bark_num)
    loss_perceptual_entropy = PerceptualEntropy(bark_num, sf, fs, win_len, psd_dict)

    # the perceptual entropy is computed batch-wise
    log_magnitude = torch.unsqueeze(torch.Tensor(log_magnitude.T), 0)
    real = torch.unsqueeze(torch.Tensor(phase.real.T), 0)
    imag = torch.unsqueeze(torch.Tensor(phase.imag.T), 0)
    x1 = torch.tensor(0.3)
    x1.requires_grad_(True)

    # compute perceptual entropy
    loss_pe = loss_perceptual_entropy(log_magnitude * x1, real, imag)

    loss_pe.backward()
    print(loss_pe)
    print(x1.grad)


if __name__ == "__main__":
    _test_perceptual_entropy(
        "/data1/gs/jiatong/SVS_system/SVS/data/public_dataset"
        "/kiritan_data/wav_info/train/1/0001.wav"
    )
