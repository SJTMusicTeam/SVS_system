#!/usr/bin/env python3
# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import argparse
import librosa
import numpy as np
import os
import pyworld as pw
import soundfile as sf


def pack_zero(number, length=4):
    number = str(number)
    return "0" * (length - len(number)) + number


def same_split(alignment):
    size = 2
    while len(alignment) / size > 330:
        size += 1
    segments = []
    start = 0
    for i in range(size - 1):
        index = round(len(alignment) / size) * (i + 1)
        while index < len(alignment) and alignment[index] != alignment[index + 1]:
            index += 1
        segments.append(alignment[start:index])
        start = index + 1
    segments.append(alignment[start:])
    return segments, size


def make_segment(alignment, sil="pau"):
    segment_info = {}
    start_id = 1
    silence_start = []
    silence_end = []
    for i in range(len(alignment)):
        if len(silence_start) == len(silence_end) and alignment[i] == sil:
            silence_start.append(i)
        elif len(silence_start) != len(silence_end) and alignment[i] != sil:
            silence_end.append(i)
        else:
            continue
    if len(silence_start) != len(silence_end):
        silence_end.append(len(alignment) - 1)
    if silence_start[0] != 0:
        if silence_end[0] - silence_start[0] > 5:
            segment_info[pack_zero(start_id)] = {
                "alignment": alignment[: silence_start[0] + 5],
                "start": 0,
            }
        else:
            segment_info[pack_zero(start_id)] = {
                "alignment": alignment[: silence_end[0]],
                "start": 0,
            }
        start_id += 1

    for i in range(len(silence_start) - 1):
        if silence_end[i] - silence_start[i] > 5:
            start = silence_end[i] - 5
        else:
            start = silence_start[i]

        if silence_end[i + 1] - silence_start[i + 1] > 5:
            end = silence_start[i + 1] + 5
        else:
            end = silence_end[i + 1]

        if end - start > 450:
            segments, size = same_split(alignment[start:end])
            pre_size = 0
            for i in range(size):
                segment_info[pack_zero(start_id)] = {
                    "alignment": segments[i],
                    "start": start + pre_size,
                }
                start_id += 1
                pre_size += len(segments[i])
            continue

        segment_info[pack_zero(start_id)] = {
            "alignment": alignment[start:end],
            "start": start,
        }
        start_id += 1

    if silence_end[-1] != len(alignment) - 1:
        if silence_end[-1] - silence_start[-1] > 5:
            segment_info[pack_zero(start_id)] = {
                "alignment": alignment[silence_end[-1] - 5 :],
                "start": silence_end[-1] - 5,
            }
        else:
            segment_info[pack_zero(start_id)] = {
                "alignment": alignment[silence_start[-1] :],
                "start": silence_start[-1],
            }
    return segment_info


def load_label(label_file, s_type="s", sr=48000, frame_shift=0.03, sil="pau"):
    label_data = open(label_file, "r")
    label_data = label_data.read().split("\n")
    quantized_align = []
    for label in label_data:
        label = label.split(" ")
        if len(label) < 3:
            continue
        if s_type == "s":
            length = (float(label[1]) - float(label[0])) / frame_shift
        else:
            length = (float(label[1]) - float(label[0])) / (frame_shift * 10e7)
        quantized_align.extend([label[-1]] * round(length))
    segment = make_segment(quantized_align, sil=sil)
    return segment, list(set(quantized_align))


def process(args):

    f0_max = 1100.0
    f0_min = 50.0

    frame_shift = args.shift_size / 1000

    hop_length = int(args.sr * frame_shift)

    lab_list = os.listdir(args.labdir)
    phone_set = []
    idscp = {}
    index = 1
    for lab in lab_list:
        lab_id = lab[:-4]
        idscp[lab_id] = index

        segments, phone = load_label(
            os.path.join(args.labdir, lab),
            s_type=args.label_type,
            sr=args.sr,
            frame_shift=frame_shift,
            sil=args.sil,
        )

        for p in phone:
            if p not in phone_set:
                phone_set.append(p)

        wav_path = os.path.join(args.wavdir, lab_id + "." + args.wav_extention)
        if args.wav_extention == "raw":
            signal, osr = sf.read(
                wav_path,
                subtype="PCM_16",
                channels=1,
                samplerate=args.sr,
                endian="LITTLE",
            )
        else:
            signal, osr = librosa.load(wav_path, sr=None)

        if osr != args.sr:
            signal = librosa.resample(signal, osr, args.sr)

        song_align = os.path.join(args.outdir, "alignment")
        song_wav = os.path.join(args.outdir, "wav_info", str(index))
        song_pitch_beat = os.path.join(args.outdir, "pitch_beat_extraction", str(index))

        if not os.path.exists(song_align):
            os.makedirs(song_align)
        if not os.path.exists(song_wav):
            os.makedirs(song_wav)
        if not os.path.exists(song_pitch_beat):
            os.makedirs(song_pitch_beat)
        print("processing {}".format(song_wav))
        for seg in segments.keys():
            alignment = segments[seg]["alignment"]
            start = segments[seg]["start"]
            name = seg
            seg_signal = signal[
                int(start * hop_length) : int(
                    start * hop_length + len(alignment) * hop_length
                )
            ]

            """extract beats"""
            tempo, beats = librosa.beat.beat_track(
                y=seg_signal, sr=args.sr, hop_length=hop_length
            )
            # times = librosa.frames_to_time(beats, sr=args.sr)
            # frames = librosa.time_to_frames(
            #     times, sr=args.sr, hop_length=hop_length, n_fft=n_fft
            # )
            np.save(os.path.join(song_pitch_beat, name) + "_beats", np.array(beats))

            """extract pitch"""
            seg_signal = seg_signal.astype("double")
            _f0, t = pw.harvest(
                seg_signal,
                args.sr,
                f0_floor=f0_min,
                f0_ceil=f0_max,
                frame_period=frame_shift * 1000,
            )
            _f0 = pw.stonemask(seg_signal, _f0, t, args.sr)

            np.save(os.path.join(song_pitch_beat, name) + "_pitch", np.array(_f0))

            alignment_id = np.zeros((len(alignment)))
            for i in range(len(alignment)):
                alignment_id[i] = phone_set.index(alignment[i])
            np.save(
                os.path.join(song_align, pack_zero(index) + name),
                np.array(alignment_id),
            )

            sf.write(
                os.path.join(song_wav, name) + ".wav", seg_signal, samplerate=args.sr
            )
            print("saved {}".format(os.path.join(song_wav, name) + ".wav"))
        index += 1

    with open(os.path.join(args.outdir, "phone_set.txt"), "w") as f:
        for p_id, p in enumerate(phone_set):
            f.write(str(p_id) + " " + p)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wavdir", type=str, help="wav data directory")
    parser.add_argument("labdir", type=str, help="label data directory")
    parser.add_argument("outdir", type=str, help="output directory")
    parser.add_argument(
        "--window_size", type=int, default=50, help="window size in miliseconds"
    )
    parser.add_argument(
        "--shift_size", type=float, default=12.5, help="shift size in miliseconds"
    )
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--sil", type=str, default="pau")
    parser.add_argument(
        "--label_type",
        type=str,
        default="s",
        help="label resolution - sample based or second based",
    )
    parser.add_argument("--label_extention", type=str, default=".txt")
    parser.add_argument("--wav_extention", type=str, default="wav")
    args = parser.parse_args()
    process(args)
