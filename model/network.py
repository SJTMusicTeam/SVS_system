#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi, Hailan Lin)


import torch
import torch.nn as nn
import numpy as np
import math
import model.module as module

class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, phone_size, embed_size, hidden_size, num_layers=1, dropout, glu_kernel=3, GLU_num):
        """
        :param para: dictionary that contains all parameters
        """
        super(Encoder, self).__init__()
        
        self.emb_phone = nn.Embedding(phone_size, embed_size)
        #full connected
        self.fc_1 = nn.Linear(embed_size, hidden_size)
        
        self.GLU_list = nn.ModuleList()
        for i in range(GLU_num):
            self.GLU_list.append(module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size))
        #self.GLU = module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)
        
        self.fc_2 = nn.Linear(hidden_size, embed_size)
        

    def forward(self, input):
        """
        input dim: [batch_size, text_phone_length]
        output dim : [batch_size, text_phone_length, embedded_dim]
        """
        embedded_phone = self.emb_phone(input)
        glu_data = self.fc_1(embedded_phone)
        
        '''这里记得改一下！'''
        for glu in enumerate(self.GLU_list):
            glu_data = glu(glu_data)
        
        glu_out = self.fc_2(torch.transpose(glu_data, 1, 2))
        
        out = embedded_phone + glu_out
        
        out = out * math.sqrt(0.5)
        return out

class Encoder_Postnet(nn.Module):
"""
Encoder Postnet
"""
def __init__(self, embedded_dim):
    super(Encoder_Postnet, self).__init__()
    
    self.fc_pitch = nn.Linear(1, embedded_dim)
    self.fc_pos = nn.Linear(1, embedded_dim)
    self.fc_beats = nn.Linear(1, embedded_dim)
    
def aligner(self, encoder_out, align_phone, text_phone):
    '''
    align_phone = [batch_size, align_phone_length]
    text_phone = [batch_size, text_phone_length]
    align_phone_length( = frame_num) > text_phone_length
    '''
    for i in range(align_phone.shape[0]):
        before_text_phone = 0
        encoder_ind = 0
        for j in range(align_phone.shape[1]):
            if align_phone[i][j] == before_text_phone:
                temp = encoder_out[i][encoder_ind]
                line = torch.cat((line,temp.unsqueeze(0)),dim = 0)
            else:
                if j == 0:
                    line = encoder_out[i][encoder_ind].unsqueeze(0)
                    before_text_phone = text_phone[i][j]
                else:
                    encoder_ind += 1
                    before_text_phone = text_phone[i][encoder_ind]
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line,temp.unsqueeze(0)),dim = 0)
        if i == 0:
            out = line.unsqueeze(0)
        else:
            out = torch.cat((out,line.unsqueeze(0)),dim = 0)

    return out
     
def forward(self, encoder_out, align_phone, text_phone, pitch, beats):
    """
    pitch/beats : [batch_size, frame_num] -> [batch_size, frame_num，1]
    """
    batch_size = pitch.shape[0]
    frame_num = pitch.shape[1]
    embedded_dim = encoder_out.shape[2]
    
    aligner_out = self.aligner(encoder_out, align_phone, text_phone)
    
    #pitch = pitch.type(torch.LongTensor)
    #print(aligner_out, pitch, pitch.type())
    pitch = self.fc_pitch(pitch.unsqueeze(0))
    #print(pitch, type(pitch))
    out = aligner_out + pitch
    
    beats = self.fc_beats(beats.unsqueeze(0))
    out = out + beats
    
    pos = module.PositionalEncoding(embedded_dim)
    pos_out = self.fc_pos(pos(torch.transpose(aligner_out, 0, 1)))
    out = out + torch.transpose(pos_out,0,1)
    
    return out
    

class Decoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self, num_block, hidden_size, output_dim, nhead=4, dropout=0.1, activation="relu",
        glu_kernel=3):
        super(Decoder, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.TransformerGLULayer(hidden_size, nhead, dropout, activation,
            glu_kernel)
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size=hidden_size

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_norm(src)
        memory, att_weght = self.decoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.output_fc(memory)
        return output

class GLU_Transformer(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, phone_size, embed_size, hidden_size, glu_num_layers, dropout, dec_num_block,
                 dec_nhead, output_dim):
        super(GLU_Transformer, self).__init__()
        self.encoder = Encoder(phone_size, embed_size, hidden_size, glu_num_layers, dropout)
        self.enc_postnet = Encoder_Postnet(embed_size)
        # TODO: standard input arguments
        self.decoder = Decoder(dec_num_block, embed_size, output_dim, dec_nhead, dropout)
        self.postnet = module.PostNet(output_dim, output_dim, output_dim)

    def forward(self, characters, phone, pitch, beat, pos_text=True, src_key_padding_mask=None,
                char_key_padding_mask=None):
        # TODO add encoder and encoder postnet
        memory = self.encoder(characters, pos=pos_text)
        mel_output, att_weight = self.decoder(memory)
        mel_output = self.postnet(mel_output)
        return mel_output


def create_src_key_padding_mask(src_len, max_len):
    bs = len(src_len)
    mask = np.zeros((bs, max_len))
    for i in range(bs):
        mask[i, src_len[i]:] = 1
    return torch.from_numpy(mask).float()


def _test():
    # debug test
    import random
    random.seed(7)
    batch_size = 16
    max_length = 500
    char_max_length = 50
    feat_dim = 1324
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_len_list = []
    for i in range(batch_size):
        seq_len_list.append(random.randint(0, max_length))

    char_seq_len_list = []
    for i in range(batch_size):
        char_seq_len_list.append(random.randint(0, char_max_length))
    spec = torch.zeros(batch_size, max_length, feat_dim)
    phone = torch.zeros(batch_size, max_length, 1)
    pitch = torch.zeros(batch_size, max_length, 1)
    beat = torch.zeros(batch_size, max_length, 1)
    char = torch.zeros([batch_size, char_max_length, 1])
    for i in range(batch_size):
        length = seq_len_list[i]
        char_length = char_seq_len_list[i]
        spec[i, :length, :] = torch.randn(length, feat_dim)
        phone[i, :length, :] = torch.randn(length, 1)
        pitch[i, :length, :] = torch.randn(length, 1)
        beat[i, :length, :] = torch.randn(length, 1)
        char[i, :length, :] = torch.randn(char_length, 1)

    seq_len = torch.from_numpy(np.array(seq_len_list)).to(device)
    char_seq_len = torch.from_numpy(np.array(char_seq_len_list)).to(device)
    spec = torch.from_numpy(spec).to(device)
    phone = torch.from_numpy(phone).to(device)
    pitch = torch.from_numpy(pitch).to(device)
    beat = torch.from_numpy(beat).to(device)

    model = GLU_Transformer(67, 256, 256, 1, 0.1, 3, 4, 10)

    spec_pred = model(char, phone, pitch, beat)



