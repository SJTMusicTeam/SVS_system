"""Copyright [2020] [linhailan1].

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
# -*- coding: utf-8 -*-
# Extract beats and pitches, and save it under the same folder as the wav file


import argparse
import librosa
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str, help="data directory")
parser.add_argument("outdir", type=str, help="output directory")
parser.add_argument("model", type=str, help="model type")
args = parser.parse_args()

if args.model == "HMM":
    frame_length = 25 / 1000
    frame_shift = 10 / 1000
elif args.model == "TDNN":
    frame_length = 60 / 1000
    frame_shift = 30 / 1000

for root, dirs, files in os.walk(args.datadir):
    for f in files:
        name, suffix = f.split(".")
        if suffix == "wav":
            y, sr = librosa.load(os.path.join(root, f), sr=None)
            hop_length = int(sr * frame_shift)
            win_length = int(sr * frame_length)
            n_fft = win_length

            """extract beats"""
            tempo, beats = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=hop_length, win_length=win_length
            )
            times = librosa.frames_to_time(beats, sr=sr)
            frames = librosa.time_to_frames(
                times, sr=sr, hop_length=hop_length, n_fft=n_fft, win_length=win_length
            )
            # file=open((os.path.join(args.outdir, name))+'_beats.txt', "w+")
            # for beat in beats:
            #    file.write(str(beat)+' ')
            # file.close()
            np.save((os.path.join(args.outdir, name)) + "_beats", np.array(beats))

            """extract pitches"""
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )
            pitches = pitches.T
            # file=open((os.path.join(args.outdir, name))+'_pitches.txt',"w+")
            pitch = np.zeros((pitches.shape[0]))
            for i in range(pitches.shape[0]):
                pitch[i] = max(pitches[i])
            # file.close()
            np.save((os.path.join(args.outdir, name)) + "_pitch", pitch)
