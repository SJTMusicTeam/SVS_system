#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.model_selection import train_test_split
import shutil
import argparse

def divide(input_path, output_path):
    train_path = output_path+'/train'
    develop_path = output_path+'/develop'
    test_path = output_path+'/test'
    
    if not os.path.exists(output_path):  
        os.mkdir(output_path)
        os.mkdir(train_path)
        os.mkdir(develop_path)
        os.mkdir(test_path)
        
    dirs_path = []
    for root, dirs, files in os.walk(input_path):
        for d in dirs: 
            dirs_path.append(d)
    
    train, test = train_test_split(dirs_path, test_size = 0.1)
    develop, test = train_test_split(test, test_size = 0.5)
    #print(train,develop,test)
    
    for d in train:
        song_path = os.path.join(train_path,d)
        wav_path = os.path.join(input_path,d)
        print(song_path,wav_path)
        if not os.path.exists(song_path):
            os.mkdir(song_path)
        for root, dirs, files in os.walk(wav_path):
            for f in files:
                if '.txt' not in f:
                    shutil.copyfile(os.path.join(wav_path,f), os.path.join(song_path, f))
    
    for d in develop:
        song_path = os.path.join(develop_path,d)
        wav_path = os.path.join(input_path,d)
        print(song_path,wav_path)
        if not os.path.exists(song_path):
            os.mkdir(song_path)
        for root, dirs, files in os.walk(wav_path):
            for f in files:
                if '.txt' not in f:
                    shutil.copyfile(os.path.join(wav_path,f), os.path.join(song_path, f))
                    
    for d in test:
        song_path = os.path.join(test_path,d)
        wav_path = os.path.join(input_path,d)
        #print(song_path,wav_path)
        if not os.path.exists(song_path):
            os.mkdir(song_path)
        for root, dirs, files in os.walk(wav_path):
            for f in files:
                if '.txt' not in f:
                    shutil.copyfile(os.path.join(wav_path,f), os.path.join(song_path, f))
                    
def dataset_split(split_output, wav_path):
    if not os.path.exists(split_output):  
        os.mkdir(split_output)
    divide(os.path.join(wav_path,'clean'), os.path.join(split_output,'clean_div'))
    divide(os.path.join(wav_path,'other'), os.path.join(split_output,'other_div'))
    mixture_divide(os.path.join(split_output,'clean_div'),os.path.join(split_output,'other_div'),os.path.join(split_output,'mixture_div'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str, help="input wav path")
    parser.add_argument("output_path", type=str, help="output directory path")
    args = parser.parse_args()
    
    #dataset_split('exp', 'wav')
    dataset_split(args.wav_path, args.output_path)

