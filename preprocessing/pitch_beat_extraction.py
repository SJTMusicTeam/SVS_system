#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:34:53 2020

@author: linhailan1
只有一首歌的beats提取，pitch提取尚未完成
"""



import librosa

root = '/Users/linhailan1/Desktop/林海斓/SVS/data_wave/clean/'
name = '青梅竹马'
path = root + name + '.txt'
song = root + name + '.wav'
file = open(path,'w+')
y, sr = librosa.load(song)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
for beat in beats:
    file.write(str(beat)+' ')

file.close()

