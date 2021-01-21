"""Copyright [2020] [Source code for nnmnkwii.metrics].

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
# Copyright @Source code for nnmnkwii.metrics
# https://r9y9.github.io/nnmnkwii/v0.0.16/_modules/nnmnkwii/metrics.html
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import librosa
import math
import numpy as np
import pyworld as pw
import scipy.fftpack
from scipy import signal
from scipy.stats.stats import pearsonr


_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)


# should work on torch and numpy arrays
def _sqrt(x):
    """_sqrt."""
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()


def _exp(x):
    """_exp."""
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.exp(x) if isnumpy else math.exp(x) if isscalar else x.exp()


def _sum(x):
    """_sum."""
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.sum(x)
    return float(x.sum())


def _linearSpec_2_mfcc(output, args):
    """_linearSpec_2_mfcc."""
    magnitude_spectrum = np.abs(output)
    power_spectrum = np.square(magnitude_spectrum)
    mel_basis = librosa.filters.mel(
        sr=args.sampling_rate,
        n_fft=args.nfft,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm=1,
    )
    mel_spectrogram = np.dot(
        power_spectrum, mel_basis.transpose(1, 0)
    )  # # dim = [batch_size, n_frames, 125]
    log_mel_spectrogram = librosa.core.spectrum.power_to_db(
        mel_spectrogram, ref=1.0, amin=1e-10, top_db=80.0
    )
    mfcc = scipy.fftpack.dct(log_mel_spectrogram, type=2, norm="ortho")[
        :, :25
    ]  # # dim = [batch_size, n_frames, 25]
    return mfcc


def Calculate_melcd_fromLinearSpectrum(output, spec, length, args):
    """Calculate_melcd_fromLinearSpectrum."""
    # output = [batch size, num frames, feat_dim]
    batch_size, num_frames, feat_dim = np.shape(output)
    output = output.cpu().detach().numpy()
    spec = spec.cpu().detach().numpy()
    length = np.max(length.cpu().detach().numpy(), axis=1)

    mfcc_predict = np.zeros([batch_size, num_frames, 25])
    mfcc_ground_truth = np.zeros([batch_size, num_frames, 25])
    mcd_total = 0
    for i in range(batch_size):
        output_nopadding = output[i][: length[i]]
        spec_nopadding = spec[i][: length[i]]

        mfcc_predict[i][: length[i]] = _linearSpec_2_mfcc(
            output_nopadding, args
        )  # # dim = [n_frames, 25]
        mfcc_ground_truth[i][: length[i]] = _linearSpec_2_mfcc(spec_nopadding, args)

        mcd_average = melcd(
            _linearSpec_2_mfcc(output_nopadding, args),
            _linearSpec_2_mfcc(spec_nopadding, args),
            [length[i]],
        )
        mcd_total += mcd_average * length[i]
    mcd_value = mcd_total / np.sum(length)

    return mcd_value, np.sum(length)


def melcd(X, Y, lengths=None):
    """Mel-cepstrum distortion (MCD).

    The function computes MCD for time-aligned mel-cepstrum sequences.

    Args:
        X (ndarray): Input mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean mel-cepstrum distortion in dB.

    .. note::

        The function doesn't check if inputs are actually mel-cepstrum.
    """
    # summing against feature axis, and then take mean against time axis
    # Eq. (1a)
    # https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf

    # X,Y = [batch size, num frames, n_mels dimension]

    if lengths is None:
        z = X - Y
        r = _sqrt((z * z).sum(-1))
        if not np.isscalar(r):
            r = r.mean()
        return _logdb_const * float(r)

    # Case for 1-dim features.
    if len(X.shape) == 2:
        # Add feature axis
        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0
    T = _sum(lengths)
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += _sqrt((z * z).sum(-1)).sum()

    return _logdb_const * float(s) / float(T)


def mean_squared_error(X, Y, lengths=None):
    """Mean squared error (MSE).

    Args:
        X (ndarray): Input features, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target features, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean squared error.

    .. tip::

        The function supports 3D padded inputs, while
        :func:`sklearn.metrics.mean_squared_error` doesn't support.
    """
    if lengths is None:
        z = X - Y
        return math.sqrt(float((z * z).mean()))

    T = _sum(lengths) * X.shape[-1]
    s = 0.0
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += (z * z).sum()

    return math.sqrt(float(s) / float(T))


def lf0_mean_squared_error(
    src_f0, src_vuv, tgt_f0, tgt_vuv, lengths=None, linear_domain=False
):
    """Mean squared error (MSE) for log-F0 sequences.

    MSE is computed for voiced segments.

    Args:
      src_f0 (ndarray): Input log-F0 sequences, shape can be either of
        (``T``,), (``B x T``) or (``B x T x 1``). Both Numpy and torch arrays
        are supported.
      src_vuv (ndarray): Input voiced/unvoiced flag array, shape can be either
        of (``T``, ), (``B x T``) or (``B x T x 1``).
      tgt_f0 (ndarray): Target log-F0 sequences, shape can be either of
        (``T``,), (``B x T``) or (``B x T x 1``). Both Numpy and torch arrays
        are supported.
      tgt_vuv (ndarray): Target voiced/unvoiced flag array, shape can be
        either of (``T``, ), (``B x T``) or (``B x T x 1``).
      lengths (list): Lengths of padded inputs. This should only be specified
        if you give mini-batch inputs.
      linear_domain (bool): Whether computes MSE on linear frequecy domain or
        log-frequency domain.

    Returns:
        float: mean squared error.
    """
    if lengths is None:
        voiced_indices = (src_vuv + tgt_vuv) >= 2
        x = src_f0[voiced_indices]
        y = tgt_f0[voiced_indices]
        if linear_domain:
            x, y = _exp(x), _exp(y)
        return mean_squared_error(x, y)

    T = 0
    s = 0.0
    for x, x_vuv, y, y_vuv, length in zip(src_f0, src_vuv, tgt_f0, tgt_vuv, lengths):
        x, x_vuv = x[:length], x_vuv[:length]
        y, y_vuv = y[:length], y_vuv[:length]
        voiced_indices = (x_vuv + y_vuv) >= 2
        T += voiced_indices.sum()
        x, y = x[voiced_indices], y[voiced_indices]
        if linear_domain:
            x, y = _exp(x), _exp(y)
        z = x - y
        s += (z * z).sum()

    return math.sqrt(float(s) / float(T))


def compute_vuv_error(src_vuv, tgt_vuv, lengths=None):
    """Voice/unvoiced error rate computation.

    Args:
        src_vuv (ndarray): Input voiced/unvoiced flag array shape
          can be either of (``T``, ), (``B x T``) or (``B x T x 1``).
        tgt_vuv (ndarray): Target voiced/unvoiced flag array shape
          can be either of (``T``, ), (``B x T``) or (``B x T x 1``).
        lengths (list): Lengths of padded inputs. This should only
          be specified if you give mini-batch inputs.

    Returns:
        float: voiced/unvoiced error rate (0 ~ 1).
    """
    if lengths is None:
        T = np.prod(src_vuv.shape)
        return float((src_vuv != tgt_vuv).sum()) / float(T)

    T = _sum(lengths)
    s = 0.0
    for x, y, length in zip(src_vuv, tgt_vuv, lengths):
        x, y = x[:length], y[:length]
        s += (x != y).sum()
    return float(s) / float(T)


def compute_f0_mse(ref_data, gen_data):
    """compute_f0_mse."""
    ref_vuv_vector = np.zeros((ref_data.size, 1))
    gen_vuv_vector = np.zeros((ref_data.size, 1))

    ref_vuv_vector[ref_data > 0.0] = 1.0
    gen_vuv_vector[gen_data > 0.0] = 1.0

    sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
    voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
    voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
    voiced_frame_number = voiced_gen_data.size

    f0_mse = (voiced_ref_data - voiced_gen_data) ** 2
    f0_mse = np.sum((f0_mse))

    # vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
    vuv_error = np.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

    return f0_mse, vuv_error, voiced_frame_number


def compute_corr(ref_data, gen_data):
    """compute_corr."""
    corr_coef = pearsonr(ref_data, gen_data)

    return corr_coef[0]


def compute_f0_corr(ref_data, gen_data):
    """compute_f0_corr."""
    ref_vuv_vector = np.zeros((ref_data.size, 1))
    gen_vuv_vector = np.zeros((ref_data.size, 1))

    ref_vuv_vector[ref_data > 0.0] = 1.0
    gen_vuv_vector[gen_data > 0.0] = 1.0

    sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
    voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
    voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
    f0_corr = compute_corr(voiced_ref_data, voiced_gen_data)

    return f0_corr


def F0_VUV_distortion(reference_list, generation_list):
    """Calculate F0-Vuv distortion.

    reference_list: ground_truth_list
    generation_list: synthesis_list
    """
    number = len(reference_list)
    total_voiced_frame_number = 0
    vuv_error = 0
    distortion = 0.0
    total_frame_number = 0

    ref_all_files_data = np.reshape(np.array([]), (-1, 1))
    gen_all_files_data = np.reshape(np.array([]), (-1, 1))

    for i in range(number):
        length1 = len(reference_list[i])
        length2 = len(generation_list[i])
        length = min(length1, length2)
        f0_ref = reference_list[i][:length].reshape(length, 1)
        f0_gen = generation_list[i][:length].reshape(length, 1)

        temp_distortion, temp_vuv_error, voiced_frame_number = compute_f0_mse(
            f0_ref, f0_gen
        )
        vuv_error += temp_vuv_error
        total_voiced_frame_number += voiced_frame_number
        distortion += temp_distortion
        total_frame_number += length

        ref_all_files_data = np.concatenate((ref_all_files_data, f0_ref), axis=0)
        gen_all_files_data = np.concatenate((gen_all_files_data, f0_gen), axis=0)

    distortion /= float(total_voiced_frame_number)
    # f0_rmse = np.sqrt(distortion)
    vuv_error /= float(total_frame_number)
    # f0_corr = compute_f0_corr(ref_all_files_data, gen_all_files_data)
    return (
        distortion,
        total_voiced_frame_number,
        vuv_error,
        total_frame_number,
        ref_all_files_data,
        gen_all_files_data,
    )


def F0_detection_wav(wav_path, signal, args):
    """F0_detection_wav."""
    f0_max = 1100.0
    f0_min = 50.0
    frame_shift = 30 / 1000

    if wav_path is not None:
        signal, osr = librosa.load(wav_path, sr=None)
    else:
        osr = args.sampling_rate
    seg_signal = signal.astype("double")
    _f0, t = pw.harvest(
        seg_signal,
        osr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=frame_shift * 1000,
    )
    _f0 = pw.stonemask(seg_signal, _f0, t, osr)

    return _f0


def invert_spectrogram(spectrogram, win_length, hop_length):
    """Apply inverse fft.

    Args:
      spectrogram: [1+n_fft//2, t]
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def griffin_lim(spectrogram, iter_vocoder, n_fft, hop_length, win_length):
    """Apply Griffin-Lim's raw."""
    X_best = copy.deepcopy(spectrogram)
    for i in range(iter_vocoder):
        X_t = invert_spectrogram(X_best, win_length, hop_length)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, win_length, hop_length)
    y = np.real(X_t)
    return y


def spectrogram2wav(
    mag, max_db, ref_db, preemphasis, power, sr, hop_length, win_length, n_fft
):
    """Generate wave file from linear magnitude spectrogram.

    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    """
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
    wav = griffin_lim(mag ** power, 100, n_fft, hop_length, win_length)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def Calculate_f0RMSE_VUV_CORR_fromWav(output, spec, length, args, flag):
    """Calculate_f0RMSE_VUV_CORR_fromWav."""
    # output = [batch size, num frames, feat_dim]
    batch_size, num_frames, feat_dim = np.shape(output)
    output = output.cpu().detach().numpy()
    spec = spec.cpu().detach().numpy()
    length = np.max(length.cpu().detach().numpy(), axis=1)

    f0_synthesis_list, f0_ground_truth_list = [], []
    for i in range(batch_size):
        output_nopadding = output[i][: length[i]]
        spec_nopadding = spec[i][: length[i]]

        wav_predict = spectrogram2wav(
            output_nopadding,
            args.max_db,
            args.ref_db,
            args.preemphasis,
            args.power,
            args.sampling_rate,
            args.frame_shift,
            args.frame_length,
            args.nfft,
        )
        wav_ground_truth = spectrogram2wav(
            spec_nopadding,
            args.max_db,
            args.ref_db,
            args.preemphasis,
            args.power,
            args.sampling_rate,
            args.frame_shift,
            args.frame_length,
            args.nfft,
        )

        f0_synthesis = F0_detection_wav(None, wav_predict, args)
        f0_ground_truth = F0_detection_wav(None, wav_ground_truth, args)

        f0_synthesis_list.append(f0_synthesis)
        f0_ground_truth_list.append(f0_ground_truth)
    (
        distortion,
        total_voiced_frame_number,
        vuv_error,
        total_frame_number,
        ref_all_files_data,
        gen_all_files_data,
    ) = F0_VUV_distortion(f0_ground_truth_list, f0_synthesis_list)

    if flag == "train":
        return (
            distortion,
            total_voiced_frame_number,
            vuv_error,
            total_frame_number,
            None,
            None,
        )
    else:
        return (
            distortion,
            total_voiced_frame_number,
            vuv_error,
            total_frame_number,
            ref_all_files_data,
            gen_all_files_data,
        )


if __name__ == "__main__":

    y = np.random.random((30000000,))
    # print(y, np.shape(y))

    y_mfcc = librosa.feature.mfcc(y=y, n_mfcc=36)
    # print(y_mfcc, np.shape(y_mfcc))

    mel_spectrogram = librosa.feature.melspectrogram(y=y)

    log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram)

    melCepstrum1 = scipy.fftpack.dct(log_mel_spectrogram, axis=0, type=2, norm="ortho")

    y = np.random.random((30,))
    # print(y, np.shape(y))

    y_mfcc = librosa.feature.mfcc(y=y, n_mfcc=36)
    # print(y_mfcc, np.shape(y_mfcc))

    mel_spectrogram = librosa.feature.melspectrogram(y=y)
    print(mel_spectrogram, np.shape(mel_spectrogram))  # dim = [n_mels, n_frames]

    log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram)
    print(
        log_mel_spectrogram, np.shape(log_mel_spectrogram)
    )  # dim = [n_mels, n_frames]

    melCepstrum2 = scipy.fftpack.dct(log_mel_spectrogram, axis=0, type=2, norm="ortho")
    print(melCepstrum2, np.shape(melCepstrum2))  # dim = [n_mels, n_frames]

    res = melcd(melCepstrum1.transpose(1, 0), melCepstrum2.transpose(1, 0))
    print(res)

    path_synthesis = "40.wav"
    path_ground_truth = "40_true.wav"

    f0_ground_truth = F0_detection_wav(path_ground_truth)

    f0_synthesis = F0_detection_wav(path_synthesis)

    f0_rmse, vuv_error, f0_corr = F0_VUV_distortion([f0_ground_truth], [f0_synthesis])
    print("with raw_f0:")
    print(
        "F0:- RMSE: {:.4f} Hz; CORR:{:.4f}; VUV: {:.4f}%".format(
            f0_rmse, f0_corr, vuv_error * 100.0
        )
    )
