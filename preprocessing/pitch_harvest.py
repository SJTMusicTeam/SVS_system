#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:34:53 2020

@author: linhailan1

Extract beats and pitches, using harvest algorithm
"""


import argparse
import librosa
import os
import numpy as np
import pyworld as pw

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str, help="data directory")
parser.add_argument("outdir", type=str, help="output directory")
#parser.add_argument("model", type=str, help="model type")
args = parser.parse_args()
'''
if args.model == 'HMM':
    frame_length = 25/1000
    frame_shift = 10/1000
elif args.model == 'TDNN':
    frame_length = 60/1000
    frame_shift = 30/1000
'''
f0_min = 50.0
f0_max = 1100.0
for root, dirs, files in os.walk(args.datadir):
    for f in files:
        name, suffix = f.split(".")
        if suffix == "wav":
            y, sr = librosa.load(os.path.join(root, f),sr = None)

            '''extract pitches'''
            #使用harvest算法计算音频的基频F0
            _f0, t = pw.harvest(y, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=pw.default_frame_period)
            _f0 = pw.stonemask(y, _f0, t, sr)
            np.save((os.path.join(args.outdir, name))+'_pitch',_f0)


            file = open((os.path.join(args.outdir, name))+'_pitch.txt', "w+")
            for f in _f0:
                file.write(str(f)+' ')
            file.close()












