import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp
import numpy as np
import math
import glu

class Encoder(nn.Module):
"""
Encoder Network
"""
def __init__(self, para):
    """
    :param para: dictionary that contains all parameters
    """
    super(Encoder, self).__init__()
    #self.alpha = nn.Parameter(t.ones(1))
    
    self.emb_phone = nn.Embedding(para['phone_size'], para['emb_dim'])
    #full connected
    self.fc_1 = nn.Linear(para['emb_dim'], para['GLU_in_dim'])
    
    self.GLU = glu.GLU(para['num_layers'], para['hidden_size'], para['kernel_size'], para['dropout'], para['GLU_in_dim'])
    
    self.fc_2 = nn.Linear(para['hidden_size'], para['emb_dim'])
    
def refine(self, align_phone):
    '''filter silence phone and repeat phone'''
    out = []
    length = []
    batch_size = align_phone.shape[0]
    max_length = align_phone.shape[1]
    before = 0
    for i in range(batch_size):
        line = []
        for j in range(max_length):
            if align_phone[i][j] == 1 or align_phone[i][j] == 0:      #silence phone or padding
                continue
            elif align_phone[i][j] == before:   #the same with the former phone
                continue
            else:
                before = align_phone[i][j]
                line.append(before)
        out.append(line)
        length.append(len(line))
    
    #pad 0
    seq_length = max(length)
    Data = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        for j in range(seq_length):
            if j < len(out[i]):
                Data[i][j] = out[i][j]
                
    return torch.from_numpy(Data).type(torch.LongTensor)
    
def forward(self, input):
    """
    input dim: [batch_size, text_phone_length]
    output dim : [batch_size, text_phone_length, embedded_dim]
    """
    input = self.refine(input)
    
    embedded_phone = self.emb_phone(input)    # [src len, batch size, emb dim]
    
    glu_out = self.GLU(self.fc_1(embedded_phone))
    
    glu_out = self.fc_2(torch.transpose(glu_out, 1, 2))
    
    out = embedded_phone + glu_out
    
    out = out *  math.sqrt(0.5)
    return out

class Encoder_Postnet(nn.Module):
    """
    Encoder Postnet
    """
    def __init__(self):
        super(Encoder_Postnet, self, seq_length).__init__()
        #length of sequence = number of frames
        self.fc = nn.Linear(seq_length, seq_length)
         
    def aligner(encoder_out, align_phone)
        return
        
    def forward(self, encoder_out, align_phone, pitch):
        aligner_out = aligner(encoder_out, align_phone)
        pitch = self.fc(pitch)
        out = aligner_out + pitch
        return



class Decoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        return

class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = Decoder()

    def forward(self, characters, mel_input, pos_text, pos_mel):
        memory, c_mask, attns_enc = self.encoder.forward(characters, pos=pos_text)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder.forward(memory, mel_input, c_mask,
                                                                                             pos=pos_mel)

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)

        return mag_pred
