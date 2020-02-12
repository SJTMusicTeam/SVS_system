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

for root, dirs, files in os.walk(args.datadir):
    for f in files:
        name, suffix = f.split(".")
        if suffix == "wav":
            file = open((os.path.join(root, name))+'.txt', "w+")
            y, sr = librosa.load(os.path.join(root, f))
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            for beat in beats:
                file.write(str(beat)+' ')
            file.close()

