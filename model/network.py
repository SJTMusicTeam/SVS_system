import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp
import numpy as np
import math
import glu
import positional_encoding

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
        '''filter repeat phone and padding'''
        out = []
        length = []
        batch_size = align_phone.shape[0]
        max_length = align_phone.shape[1]
    
        for i in range(batch_size):
            line = []
            before = 0
            for j in range(max_length):
                if align_phone[i][j] == 0:      #filter padding
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
        super(Encoder_Postnet, self, hidden_size).__init__()
        
        self.fc_pitch = nn.Linear(1, hidden_size)
        self.fc_pos = nn.Linear(1, hidden_size)
        self.fc_beats = nn.Linear(1, hidden_size)
        
    def aligner(self, encoder_out, align_phone):
        """padding 的情况还未解决"""
        out = []
        for i in range(align_phone.shape(0)):
            line = []
            before_phone = 0
            encoder_ind = 0
            for j in range(align_phone.shape(1)):
                if align_phone[i][j] == before_phone:
                    line.append(encoder_out[i][encoder_ind])
                else:
                    before_phone = align_phone[i][j]
                    encoder_ind += 1
                    line.append(encoder_out[i][encoder_ind])
            out.append(line)
            
        return torch.tensor(out)
         
    def forward(self, encoder_out, align_phone, pitch, beats):
        """
        pitch/beats : [batch_size, frame_num] -> [batch_size, frame_num，1]
        """
        batch_size = pitch.shape(0)
        frame_num = pitch.shape(1)
        embedded_dim = encoder_out.shape(2)
        
        aligner_out = aligner(encoder_out, align_phone)
        
        pitch = self.fc_pitch(torch.tensor(pitch).unsqueeze(0))
        out = aligner_out + pitch
        
        beats = self.fc_pitch(torch.tensor(beats).unsqueeze(0))
        out = out + beats
        
        pos = positional_encoding.PositionalEncoding(embedded_dim)
        pos_out = self.fc_pos(pos(torch.transpose(aligner_out, 0, 1)))
        out = out + torch.transpose(pos_out,0,1)
        
        return out



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
        super(Model, self, para).__init__()
        self.encoder = Encoder(para)
        self.enc_postnet = Encoder_Postnet(para['embedded_size'])
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
