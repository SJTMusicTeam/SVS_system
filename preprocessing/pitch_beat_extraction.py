#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:34:53 2020

@author: linhailan1

Extract beats and pitches, and save it under the same folder as the wav file
"""


import argparse
import librosa
import os

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str, help="data directory")
args = parser.parse_args()


frame_length = 25/1000
frame_shift = 10/1000
for root, dirs, files in os.walk(args.datadir):
    for f in files:
        name, suffix = f.split(".")
        if suffix == "wav":
            y, sr = librosa.load(os.path.join(root, f),sr = None)
            hop_length = int(sr * frame_shift)
            n_fft = int(sr * frame_length)
            
            '''extract beats'''
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            times = librosa.frames_to_time(beats, sr=sr)
            frames = librosa.time_to_frames(times,sr = sr, hop_length=hop_length, n_fft = n_fft)
            file = open((os.path.join(root, name))+'_beats.txt', "w+")
            for beat in beats:
                file.write(str(beat)+' ')
            file.close()
            
            '''extract pitches'''
            pitches, magnitudes = librosa.piptrack(y=y,sr=sr,n_fft=n_fft,hop_length=hop_length)
            pitches = pitches.T
            file = open((os.path.join(root, name))+'_pitches.txt', "w+")
            for p in pitches:
                file.write(str(max(p))+' ')
            file.close()















