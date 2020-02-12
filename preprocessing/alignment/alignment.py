#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:51:15 2020

@author: linhailan1
"""

import numpy as np
import math
import os
import argparse

def DTW(template, sample): 
    '''
    Alignment using DTW algorithm
    Return the record and distance of the shortest path 
    '''
    Max = -math.log(1e-300,2)     
    T_len = len(template)
    S_len = len(sample)
    before = [Max for i in range(T_len)]
    after = [Max for i in range(T_len)]
    record_b = [[] for i in range(T_len)]
    record_a = [[] for i in range(T_len)]
    
    #initial the first frame
    for key in sample[0].keys():
        if key in template:  
            ind = template.index(key)
            before[ind] = sample[0][key]
            
    for i in range(1, S_len):     
        '''
        calculate the probability of all phonemes 
        in the corresponding template in the i-th frame
        '''
        frame_pos = [Max for i in range(T_len)]
        for key in sample[i].keys():
            if key in template:  
                ind = template.index(key)
                frame_pos[ind] = sample[i][key]
                
        for j in range(T_len):
            cij = frame_pos[j]
            delta = 0
            if j >= 2:
                if before[j] < before[j-1] and before[j] < before[j-2]: 
                    delta = 0
                elif before[j-1] < before[j] and before[j-1] < before[j-2]:
                    delta = 1
                else:
                    delta = 2
            elif j >= 1:
                if before[j] < before[j-1]: 
                    delta = 0
                else:
                    delta = 1
                    
            after [j] = before[j-delta] + cij
            record_a[j] = record_b[j-delta][:]      
            record_a[j].append(template[j-delta])
            if i == S_len - 1:  record_a[j].append(template[j])
        before = after[:]
        record_b = record_a[:]
    ind = np.argmin(after)
    return record_a[ind],after[ind]

def text_to_matrix(Map,file):
    '''
    Read the posterior probability matrix and save it in dictionary M
    '''
    Min = 1e-300
    M = dict()
    post = file.readlines()
    for song in post:      #every song in file
        song = song.split(' [ ')
        name = song[0]
        line = []
        for frame in song[1:]:    #every frame in song
            frame = frame.split()
            i = 0
            pos = dict()
            while 2 * i < len(frame) - 1: #Probability of each phoneme
                ind = int(frame[2 * i])
                temp = float(frame[2 * i + 1])
                if temp < Min:  temp = Min      #Avoid taking the log of 0  
                if Map[ind] in pos.keys():   
                    temp = 2**(-pos[Map[ind]]) + temp 
                    if temp > 1: temp = 1
                    possi = -math.log(temp,2)
                    pos[Map[ind]] = possi
                else:
                    possi = -math.log(temp,2)
                    pos[Map[ind]] = possi
                i = i + 1
            line.append(pos)
        M[name] = line
    return M

def index_to_phone(file):
    '''
    Establish the correspondence between phonemes and index,
    and save them in the Map dictionary, (key = index, value = phoneme name)
    '''
    lines = file.readlines()
    Map = dict()
    for line in lines:
        if line[0] == '#':  break
        line = line.split()
        #filter number and combine the same phonemes in different tones
        temp = ''.join(list(filter(str.isalpha, line[0])))
        Map[int(line[1])] = temp
    return Map
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phone_map_path", type=str, help="phone.txt path")
    parser.add_argument("phone_post_path", type=str, help="posterior probability matrix file phone.post path")
    args = parser.parse_args()
    
    file = open(args.phone_map_path,'r')
    Map = index_to_phone(file)
    file.close()
    file = open(args.phone_post_path,'r')
    Matrix = text_to_matrix(Map,file)
    file.close()
    for name in Matrix.keys():
        template = ['a','b','c']        #此处应换成读模版（标注）文件并解析
        record,result = DTW(template,Matrix[name])
        #最后将结果写入文件
        
    '''
    单独测试DTW算法的例子
    template = ['a','b','c']
    sample = [{'a':0.1,'b':1},{'a':0.2,'b':0.9},{'a':1,'b':0.2},{'a':2,'b':0.2,'c':3},{'b':0.1,'c':2},{'b':0.3,'c':1},{'b':0.5,'c':0.4},{'b':2,'c':0.1}]
    record,result = DTW(template,sample)
    print(record)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
