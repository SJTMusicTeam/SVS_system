#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:34:53 2020

@author: linhailan1

Extract beats and save it under the same folder as the wav file
Pitch extraction haven't finished yet!
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
            file = open((os.path.join(root, name))+'.txt', "w+")
            y, sr = librosa.load(os.path.join(root, f),sr = None)
            hop_length = int(sr * frame_shift)
            n_fft = int(sr * frame_length)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            times = librosa.frames_to_time(beats, sr=sr)
            frames = librosa.time_to_frames(times,sr = sr, hop_length=hop_length, n_fft = n_fft)
            #file.write(str(beats)+'\n' + str(times)+'\n' + str(frames))
            '''
            for i in range(len(y)):
                if i in beats:
                    file.write('1 ')
                else:
                    file.write('0 ')
            '''
            for beat in frames:
                file.write(str(beat)+' ')
            
            file.close()

