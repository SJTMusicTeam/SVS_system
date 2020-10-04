# -*- coding: utf-8 -*-
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


import os

def Get_align_beat_pitch_spectrogram(align_root_path, pitch_beat_root_path, wav_root_path):
    
    filename_list = os.listdir(align_root_path) #列出文件夹下所有的目录与文件
    path_list = []
    phone_list, beat_list, pitch_list, spectrogram_list = [],[],[],[]
    
    for i in range(0,len(filename_list)):
        if filename_list[i][-1] != 'm' and filename_list[i][-1] != 'e':
            path = os.path.join(align_root_path, filename_list[i])
            path_list.append(path)
            
#            print(filename_list[i][1:4], filename_list[i][4:])
            
            with open(path, 'r') as f:
                phone = f.read().strip().split(" ")
                phone_list.append(phone)
                f.close()
            beat_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4], filename_list[i][4:]+"_beats.txt")
            with open(beat_path, 'r') as f:
                beat_list.append(f.read().strip().split(" "))
            pitch_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4], filename_list[i][4:]+"_pitches.txt")
            with open(pitch_path, 'r') as f:
                pitch_list.append(f.read().strip().split(" "))
                
            wav_path = os.path.join(wav_root_path, filename_list[i][1:4], filename_list[i][4:]+".wav")
            frame_length = 60/1000
            frame_shift = 30/1000            
            y, sr = librosa.load(wav_path,sr = None)
            hop_length = int(sr * frame_shift)
            n_fft = int(sr * frame_length)
            spectrogram_list.append(librosa.feature.melspectrogram(y=y, sr=sr,hop_length=hop_length, n_fft = n_fft))
    
    return phone_list, beat_list, pitch_list, spectrogram_list


if __name__ == "__main__":

    align_root_path = "C:/Users/PKU/Desktop/SVS_system/preprocessing/ch_asr/exp/alignment/clean_set/" #文件夹目录
    pitch_beat_root_path = "C:/Users/PKU/Desktop/SVS_system/preprocessing/ch_asr/exp/pitch_beat_extraction/clean/"
    wav_root_path = 'C:/Users/PKU/Desktop/SVS_system/annotation/clean/'
    
    phone_list, beat_list, pitch_list, spectrogram_list = Get_align_beat_pitch_spectrogram(align_root_path, pitch_beat_root_path, wav_root_path)
    
    length = []
    for i in range(len(phone_list)):
        length.append(len(phone_list[i]))
        
    sample_num = len(phone_list)
    seq_length = max(length)
    
    
    Data = np.zeros((sample_num,seq_length,3))
    Label = np.zeros((sample_num,seq_length,128))
    
    for i in range(sample_num):
        for j in range(seq_length):
            if j < len(phone_list[i]):
                Data[i][j][0] = np.array(phone_list[i][j])
            if str(j) in beat_list[i]:
                Data[i][j][1] = 1
            if j < len(phone_list[i]):  # 在这里写phone_list是因为每一个样本，pitch都比phone多一帧（原则：所有以phone为准）
                Data[i][j][2] = np.array(pitch_list[i][j])
                Label[i][j] = spectrogram_list[i][:,j]
    
    
    #创建子类
    class MyDataset(Dataset):
        #初始化，定义数据内容和标签
        def __init__(self, Data, Label):
            self.Data = Data
            self.Label = Label
        #返回数据集大小
        def __len__(self):
            return len(self.Data)
        #得到数据内容和标签
        def __getitem__(self, index):
            data = torch.Tensor(self.Data[index])
            label = torch.IntTensor(self.Label[index])
            return data, label
    
    dataset = MyDataset(Data, Label)
    # print(dataset)
    # print('dataset大小为：', dataset.__len__())
    # print(dataset.__getitem__(0))
    # print(dataset[0])
#
##创建DataLoader迭代器
    dataloader = DataLoader(dataset,batch_size= 2, shuffle = False, num_workers= 0)
    for i, item in enumerate(dataloader):
        print('i:', i)
        data, label = item
        print('data:', data)
        print('label:', label)
    
    
    
