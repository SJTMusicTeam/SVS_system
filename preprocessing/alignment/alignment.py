#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:51:15 2020

@author: linhailan1
"""

import numpy as np
import math


def DTW(template, sample): 
    '''
    使用DTW算法进行对齐
    '''
    Max = -math.log(1e-300,2)     
    T_len = len(template)
    S_len = len(sample)
    before = [Max for i in range(T_len)]
    after = [Max for i in range(T_len)]
    record_b = [[] for i in range(T_len)]
    record_a = [[] for i in range(T_len)]
    
    #初始化第0帧
    for key in sample[0].keys():
        if key in template:  
            ind = template.index(key)
            before[ind] = sample[0][key]
            
    for i in range(1, S_len):       #从1开始，到S_len-1结束
        #计算每一帧对应模版中所有音素的概率
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
            record_a[j] = record_b[j-delta][:]      #[:]是深拷贝的意思！！！
            record_a[j].append(template[j-delta])
            if i == S_len - 1:  record_a[j].append(template[j])
        before = after[:]
        record_b = record_a[:]
    ind = np.argmin(after)
    return record_a[ind],after[ind]

def text_to_matrix(Map):
    '''
    读入后验概率矩阵并保存至字典M里
    '''
    Min = 1e-300
    M = dict()
    file = open('/Users/linhailan1/Desktop/林海斓/phone.post','r')
    post = file.readlines()
    file.close()
    for song in post:      #每一首曲子
        song = song.split(' [ ')
        name = song[0]
        line = []
        for frame in song[1:]:    #每一帧
            frame = frame.split()
            i = 0
            pos = dict()
            while 2 * i < len(frame) - 1: #每个音素的概率
                ind = int(frame[2 * i])
                temp = float(frame[2 * i + 1])
                if temp < Min:  temp = Min      #避免对0取对数  
                if Map[ind] in pos.keys():   #说明之前已经存在过相同音素不同音调的音素了
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

def phone_to_index():
    '''
    建立音素与编号的对应关系，保存在Map字典里,(key = 编号，value = 音素名)
    '''
    file = open('/Users/linhailan1/Desktop/林海斓/phones.txt','r')
    lines = file.readlines()
    file.close()
    Map = dict()
    for line in lines:
        if line[0] == '#':  break
        line = line.split()
        temp = ''.join(list(filter(str.isalpha, line[0])))
        Map[int(line[1])] = temp
    return Map
        
        
if __name__ == "__main__":
    
    Map = phone_to_index()
    Matrix = text_to_matrix(Map)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
