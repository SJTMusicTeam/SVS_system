#!/usr/bin/env python
# coding: utf-8

import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import random
import math
import time
import copy

from scipy import signal
from scipy.io.wavfile import write
import hyperparams as hp

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # to decibel
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mag

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y
def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def Get_align_beat_pitch_spectrogram(align_root_path, pitch_beat_root_path, wav_root_path):
    
    filename_list = os.listdir(align_root_path) #列出文件夹下所有的目录与文件
    path_list = []
    phone_list, beat_list, pitch_list, spectrogram_list = [],[],[],[]
    
    for i in range(0,len(filename_list)):
        if filename_list[i][-1] != 'm' and filename_list[i][-1] != 'e':
            path = os.path.join(align_root_path, filename_list[i])
            path_list.append(path)
            
            with open(path, 'r') as f:
                phone = f.read().strip().split(" ")
                f.close()
            beat_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4], filename_list[i][4:]+"_beats.txt")
            with open(beat_path, 'r') as f:
                beat = f.read().strip().split(" ")
                f.close()
            pitch_path = os.path.join(pitch_beat_root_path, filename_list[i][1:4], filename_list[i][4:]+"_pitches.txt")
            with open(pitch_path, 'r') as f:
                pitch = f.read().strip().split(" ")  
                f.close()
            wav_path = os.path.join(wav_root_path, filename_list[i][1:4], filename_list[i][4:]+".wav")
            spectrogram = get_spectrograms(wav_path).T
            
            # length check
            min_length = min(len(phone), np.shape(spectrogram)[1])
            phone_list.append(phone[:min_length])
            beat_list.append(beat[:min_length])
            pitch_list.append(pitch[:min_length])
            spectrogram_list.append(spectrogram[:,:min_length])
    
    # sort by length desc
    length = []
    for i in range(len(phone_list)):
        length.append(len(phone_list[i]))

    phone_list = [x for _,x in sorted(zip(length,phone_list),reverse=True)]
    beat_list = [x for _,x in sorted(zip(length,beat_list),reverse=True)]
    pitch_list = [x for _,x in sorted(zip(length,pitch_list),reverse=True)]
    # spectrogram_list = sorted(spectrogram_list,key=lambda x:np.shape(x)[1],reverse=True)
    spectrogram_list = [x for _,x in sorted(zip(length,spectrogram_list), key=lambda x:x[0], reverse=True)]
    
    length = sorted(length,reverse=True)
    
    return phone_list, beat_list, pitch_list, spectrogram_list, length

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, device):
        super().__init__()
        
        self.embedding_phone = nn.Embedding(input_dim, emb_dim)
        self.embedding_pitch = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim+2, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src, src_len):
        
        #src = [src len, batch size, 3]
        #src_len = [batch size]
        
        src_phone = src[:,:,0]    # [src len, batch size]
        src_beat = src[:,:,1]     # [src len, batch size]
        src_pitch = src[:,:,2]    # [src len, batch size]
        
        embedded_phone = self.dropout(self.embedding_phone(src_phone.type(torch.LongTensor).to(self.device)))    # [src len, batch size, emb dim]
        # embedded_pitch = self.dropout(self.embedding_pitch(src_pitch))    # [src len, batch size, emb dim]
        
        src_beat = src_beat.unsqueeze(2)    # [src len, batch size, 1]
        src_pitch = src_pitch.unsqueeze(2)  # [src len, batch size, 1]
        
        embedded = torch.cat((embedded_phone, src_beat, src_pitch), dim = 2)   # [src len, batch size, emb_dim+2]
        
        #embedded = [src len, batch size, emb dim]
                
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  ENcoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.attention = attention
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_hid1 = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, 2048)
        self.fc_hid2 = nn.Linear(2048, 1600)
        self.fc_out = nn.Linear(1600, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size，1324]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size，1324]
        
        embedded = input
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.dropout(F.relu(self.fc_hid1(torch.cat((output, weighted, embedded), dim = 1))))
        prediction = self.dropout(F.relu(self.fc_hid2(prediction)))
        prediction = F.relu(self.fc_out(prediction))

        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        input = outputs[0]

        for t in range(0, trg_len):
            
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # check output within the len of each_sample, pad 0 as over its len 
            for each_sample in range(batch_size):
                if t < src_len[each_sample]:
                    outputs[t][each_sample] = output[each_sample]
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src, trg, src_len, max_seqLength = batch

        optimizer.zero_grad()
        
        src = src.permute(1,0,2)
        trg = trg.permute(1,0,2)
        output = model(src, src_len, trg)
        
        #trg = [trg len, batch size, 1324]
        #output = [trg len, batch size, 1324]
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
	
        if i == 0:
            output_filename = "result/output.npy"
            np.save(output_filename, output.cpu().detach().numpy())
        print("batch_i", i, max_seqLength, loss.item())

    return epoch_loss / len(iterator), output, trg

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, trg, src_len = batch
            src = src.permute(1,0,2)
            trg = trg.permute(1,0,2)
            output = model(src, src_len, trg, 0) #turn off teacher forcing
            
            #trg = [trg len, batch size, 1324]
            #output = [trg len, batch size, 1324]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), output

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class MyDataset(Dataset):
        #初始化，定义数据内容和标签
        def __init__(self, Data, Label, length, device):
            self.Data = Data
            self.Label = Label
            self.length = length
            self.device = device
        #返回数据集大小
        def __len__(self):
            return len(self.Data)
        #得到数据内容和标签
        def __getitem__(self, index):
            data = torch.FloatTensor(self.Data[index]).to(self.device)
            label = torch.FloatTensor(self.Label[index]).to(self.device)
            length = seq_lengths[index]
            return data, label, length

def collate_fn_padd(batch):
    # get data from zipped batch
    data, label, length = zip(*batch)
    
    max_length = length[0].cpu().numpy()

    # transform tuple to list, because tuple can`t change it value
    data, label = list(data), list(label)
    
    for i in range(len(data)):
        data[i] = data[i][:max_length,:]
        label[i] = label[i][:max_length,:]
    
    # # transform list to tensor array
    data, label = torch.stack(data), torch.stack(label)
    
    return data, label, length, max_length

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)

    align_root_path = "/data1/gs/SVS_system/preprocessing/ch_asr/exp/alignment/clean_set/" #文件夹目录
    pitch_beat_root_path = "/data1/gs/SVS_system/preprocessing/ch_asr/exp/pitch_beat_extraction/clean/"
    wav_root_path = '/data1/gs/annotation/clean/'
    phone_list, beat_list, pitch_list, spectrogram_list, length = Get_align_beat_pitch_spectrogram(align_root_path, pitch_beat_root_path, wav_root_path)
    
    phone_list = phone_list[270:]
    beat_list = beat_list[270:]
    pitch_list = pitch_list[270:]
    spectrogram_list = spectrogram_list[270:]
    length = length[270:]
    
    
    sample_num = len(phone_list)
    seq_length = max(length)
    seq_lengths = torch.LongTensor(length).to(device)

    Data = np.zeros((sample_num,seq_length,3))  
    Label = np.zeros((sample_num,seq_length,1324))

    for i in range(sample_num):
        for j in range(seq_length):
            if j < len(phone_list[i]):
                Data[i][j][0] = np.array(phone_list[i][j])
            if str(j) in beat_list[i]:
                Data[i][j][1] = 1
            if j < len(phone_list[i]):  # 在这里写phone_list是因为每一个样本，pitch都比phone多一帧（原则：所有以phone为准）
                Data[i][j][2] = np.array(pitch_list[i][j])
                Label[i][j] = spectrogram_list[i][:,j]


    dataset = MyDataset(Data, Label, seq_lengths, device)
    dataloader = DataLoader(dataset,
                            batch_size= 32, 
                            shuffle = False,
                            collate_fn = collate_fn_padd,
                            num_workers= 0)

    print("DataSet Prepare Succeed!")
    # for i, batch in enumerate(dataloader):
    #     if i == 3:
    #         data, label, lengths, max_length = batch
    #         print('data:', data[:,:,2])
    #         # print('label:', label)
    #         print('length:', lengths)
    #         print(type(lengths))

    INPUT_DIM = 70    # phone dim
    OUTPUT_DIM = 1324    # mel_spectrogram dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = OUTPUT_DIM
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)


    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.MSELoss(reduction = 'sum')


    N_EPOCHS = 12
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss, output, trg = train(model, dataloader, optimizer, criterion, CLIP)
        # valid_loss = evaluate(model, valid_iterator, criterion)
        output_filename = "result/output_epoch_" + str(epoch) + ".npy"
        np.save(output_filename, output.cpu().detach().numpy())	
        
        if epoch == N_EPOCHS-1:
            trg_filename = "result/trg.npy"
            np.save(trg_filename, trg.cpu().detach().numpy()) 

        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'tut4-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')







