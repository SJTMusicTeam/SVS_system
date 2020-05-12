#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi, Hailan Lin)

# debug only
# import sys
# sys.path.append("/Users/jiatongshi/projects/svs_system/SVS_system")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import model.module as module
from model.pretrain_module import clones,FFN,Attention
from torch.nn.init import xavier_uniform_
from torch.nn.init import xavier_normal_
from torch.nn.init import constant_

from model.global_mvn import GlobalMVN
class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, phone_size, embed_size, hidden_size, dropout, GLU_num, num_layers=1, glu_kernel=3):
        """
        :param para: dictionary that contains all parameters
        """
        super(Encoder, self).__init__()
        
        self.emb_phone = nn.Embedding(phone_size, embed_size)
        #full connected
        self.fc_1 = nn.Linear(embed_size, hidden_size)
        
        self.GLU_list = nn.ModuleList()
        for i in range(int(GLU_num)):
            self.GLU_list.append(module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size))
        #self.GLU = module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)
        
        self.fc_2 = nn.Linear(hidden_size, embed_size)
    

    def forward(self, text_phone, pos=None):
        """
        text_phone dim: [batch_size, text_phone_length]
        output dim : [batch_size, text_phone_length, embedded_dim]
        """
        # don't use pos in glu, but leave the field for uniform interface
        embedded_phone = self.emb_phone(text_phone)
        glu_in = self.fc_1(embedded_phone)
        
        batch_size = glu_in.shape[0]
        text_phone_length = glu_in.shape[1]
        embedded_dim = glu_in.shape[2]
        
        for glu in self.GLU_list:
            glu_out = glu(glu_in)
            glu_in = glu_out.reshape(batch_size, text_phone_length, embedded_dim)
        
        glu_out = self.fc_2(glu_in)
        
        out = embedded_phone + glu_out
        
        out = out * math.sqrt(0.5)
        return out, text_phone


class SA_Encoder(nn.Module):
    def __init__(self,phone_size, embed_size, hidden_size,dropout,num_blocks=3,nheads=4):
        super(SA_Encoder, self).__init__()
        self.layers = clones(Attention(hidden_size), int(num_blocks))
        self.ffns = clones(FFN(hidden_size), int(num_blocks))
        self.emb_phone = nn.Embedding(phone_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, hidden_size)

    def forward(self, text_phone, pos):

        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, text_phone.size(1), 1)

        embedded_phone = self.emb_phone(text_phone)
        x = self.fc_1(embedded_phone)
        attns = []
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=query_mask)
            x = ffn(x)
            attns.append(attn) 
        return x , text_phone


class Encoder_Postnet(nn.Module):
    """
    Encoder Postnet
    """
    def __init__(self, embed_size):
        super(Encoder_Postnet, self).__init__()

        self.fc_pitch = nn.Linear(1, embed_size)
        #Remember! embed_size must be even!!
        self.fc_pos = nn.Linear(embed_size, embed_size)
        #only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, embed_size)
        self.pos = module.PositionalEncoding(embed_size)

    def aligner(self, encoder_out, align_phone, text_phone):
        '''
        align_phone = [batch_size, align_phone_length]
        text_phone = [batch_size, text_phone_length]
        align_phone_length( = frame_num) > text_phone_length
        '''
        # batch
        align_phone = align_phone.long()
        for i in range(align_phone.shape[0]):
            before_text_phone = text_phone[i][0]
            encoder_ind = 0
            line = encoder_out[i][0].unsqueeze(0)
            # frame
            for j in range(1,align_phone.shape[1]):
                if align_phone[i][j] == before_text_phone:
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line,temp.unsqueeze(0)),dim = 0)
                else:
                    encoder_ind += 1
                    if encoder_ind >= text_phone[i].size()[0]:
                        break
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
       
        pitch = self.fc_pitch(pitch)
        out = aligner_out + pitch
        
        beats = self.emb_beats(beats.squeeze(2))
        out = out + beats
        
        
        pos_encode = self.pos(torch.transpose(aligner_out,0,1))
        pos_out = self.fc_pos(torch.transpose(pos_encode,0,1))
        
        out = out + pos_out
        
        return out
    
class Decoder_noGLU(nn.Module):
    """
    Decoder Network
    """
    # TODO： frame smoothing (triple the time resolution)
    def __init__(self, num_block, hidden_size, output_dim, nhead=4, dropout=0.1, activation="relu",
        glu_kernel=3, local_gaussian=False, device="cuda"):
        super(Decoder_noGLU, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.Transformer_noGLULayer(hidden_size, nhead, dropout, activation,
            glu_kernel, local_gaussian=local_gaussian, device=device)
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size=hidden_size

    def forward(self, src, pos):
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        src = self.input_norm(src)
        memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output, att_weight

   
class Decoder(nn.Module):
    """
    Decoder Network
    """
    # TODO： frame smoothing (triple the time resolution)
    def __init__(self, num_block, hidden_size, output_dim, nhead=4, dropout=0.1, activation="relu",
        glu_kernel=3, local_gaussian=False, device="cuda"):
        super(Decoder, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.TransformerGLULayer(hidden_size, nhead, dropout, activation,
            glu_kernel, local_gaussian=local_gaussian, device=device)
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size=hidden_size

    def forward(self, src, pos):
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        src = self.input_norm(src)
        memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output, att_weight

class GLU_TransformerSVS(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, phone_size, embed_size, hidden_size, glu_num_layers, dropout, dec_num_block,
                 dec_nhead, output_dim, n_mels=-1, local_gaussian=False, device="cuda"):
        super(GLU_TransformerSVS, self).__init__()
        self.encoder = Encoder(phone_size, embed_size, hidden_size, dropout, glu_num_layers,
                               num_layers=1, glu_kernel=3)
        self.enc_postnet = Encoder_Postnet(embed_size)

        self.use_mel = (n_mels > 0)

        if self.use_mel:
            self.decoder = Decoder(dec_num_block, embed_size, n_mels, dec_nhead, dropout,
                                   local_gaussian=local_gaussian, device=device)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(dec_num_block, embed_size, output_dim, dec_nhead, dropout,
                                   local_gaussian=local_gaussian, device=device)
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(self, characters, phone, pitch, beat, pos_text=True, pos_char=None,
                pos_spec=None):

        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)
        output = self.postnet(mel_output)
        return output, att_weight, mel_output
class GLU_TransformerSVS_norm(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, stats_file,stats_mel_file,phone_size, embed_size, hidden_size, \
                 glu_num_layers, dropout, dec_num_block,dec_nhead, output_dim, n_mels=-1, \
                 local_gaussian=False, device="cuda"):
        super(GLU_TransformerSVS_norm, self).__init__()
        self.encoder = Encoder(phone_size, embed_size, hidden_size, dropout, glu_num_layers,
                               num_layers=1, glu_kernel=3)
        self.enc_postnet = Encoder_Postnet(embed_size)
        self.normalizer = GlobalMVN(stats_file)
        self.mel_normalizer = GlobalMVN(stats_mel_file)

        self.use_mel = (n_mels > 0)

        if self.use_mel:
            self.decoder = Decoder(dec_num_block, embed_size, n_mels, dec_nhead, dropout,
                                   local_gaussian=local_gaussian, device=device)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(dec_num_block, embed_size, output_dim, dec_nhead, dropout,
                                   local_gaussian=local_gaussian, device=device)
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(self, spec,mel,characters, phone, pitch, beat, pos_text=True, pos_char=None,
                pos_spec=None):

        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)
        output = self.postnet(mel_output)

        spec,_=self.normalizer(spec,pos_spec)
        if mel is not None:
            mel,_=self.mel_normalizer(mel,pos_spec)
        return output, att_weight, mel_output,spec,mel



class LSTMSVS(nn.Module):
    """
    LSTM singing voice synthesis model
    """

    def __init__(self, embed_size=512, d_model=512, d_output=1324,
                 num_layers=2, phone_size=87, n_mels=-1,
                 dropout=0.1, device="cuda", use_asr_post=False):
        super(LSTMSVS, self).__init__()
        
        if use_asr_post:
            self.input_fc = nn.Linear(phone_size - 1, d_model)
        else:
            self.input_embed = nn.Embedding(phone_size, embed_size)

        self.linear_wrapper = nn.Linear(embed_size, d_model)

        self.phone_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout)

        self.linear_wrapper2 = nn.Linear(d_model * 2, d_model)

        self.pos = module.PositionalEncoding(d_model)
        #Remember! embed_size must be even!!
        assert embed_size % 2 == 0
        self.fc_pos = nn.Linear(d_model, d_model)

        self.pos_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout)

        self.fc_pitch = nn.Linear(1, d_model)
        self.linear_wrapper3 = nn.Linear(d_model * 2, d_model)
        self.pitch_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout)
        
        #only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, d_model)
        self.linear_wrapper4 = nn.Linear(d_model * 2, d_model)
        self.beats_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout)
    
        
        self.output_fc = nn.Linear(d_model * 2, d_output)

        self.use_mel = (n_mels > 0)
        self.n_mels = n_mels
        if self.use_mel:
            self.output_mel = nn.Linear(d_model * 2, n_mels)
            self.postnet = module.PostNet(n_mels, d_output, (d_output // 2 * 2))
        else:
            self.output_fc = nn.Linear(d_model * 2, d_output)


        self._reset_parameters()

        self.use_asr_post = use_asr_post
        self.d_model = d_model

    def forward(self, phone, pitch, beats):
        if self.use_asr_post:
            out = self.input_fc(phone.squeeze(-1))
        else:
            out = self.input_embed(phone.squeeze(-1))
        out = F.leaky_relu(out)
        out = F.leaky_relu(self.linear_wrapper(out))
        out, _ = self.phone_lstm(out)
        out = F.leaky_relu(self.linear_wrapper2(out))

        pos = self.pos(out)
        pos_encode = self.fc_pos(pos)
        out = pos + out
        out, _ = self.pos_lstm(out)
        out = F.leaky_relu(self.linear_wrapper3(out))
        pitch = F.leaky_relu(self.fc_pitch(pitch))
        out = pitch + out
        out, _ = self.pitch_lstm(out)
        out = F.leaky_relu(self.linear_wrapper4(out))
        beats = F.leaky_relu(self.emb_beats(beats.squeeze(-1)))
        out = beats + out
        out, (h0, c0) = self.beats_lstm(out)
        if self.use_mel:
            mel = self.output_mel(out)
            out = self.postnet(mel)
            return out, (h0, c0), mel
        else:
            out = self.output_fc(out)
            return out, (h0, c0), None

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TransformerSVS(GLU_TransformerSVS):
    def __init__(self, phone_size, embed_size, hidden_size, glu_num_layers, dropout, dec_num_block,
            dec_nhead, output_dim, n_mels=80, local_gaussian=False, device="cuda"):
        super(TransformerSVS, self).__init__(phone_size, embed_size, hidden_size,
                glu_num_layers, dropout, dec_num_block,dec_nhead, output_dim,
                local_gaussian=local_gaussian, device="cuda")
        self.encoder = SA_Encoder(phone_size, embed_size, hidden_size, dropout)
        self.use_mel = (n_mels > 0) # FIX ME
        if self.use_mel:
            self.decoder = Decoder(dec_num_block,embed_size,n_mels,dec_nhead,dropout,device=device)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim// 2*2))

class TransformerSVS_norm(nn.Module):
    def __init__(self,stats_file,stats_mel_file,phone_size,embed_size,hidden_size,glu_num_layers,dropout,\
                 dec_num_block,dec_nhead,output_dim,n_mels=80,local_gaussian=False,device="cuda"):
        super(TransformerSVS_norm,self).__init__()
        self.encoder = SA_Encoder(phone_size,embed_size,hidden_size,dropout)
        self.normalizer = GlobalMVN(stats_file)   # FIX ME, add utterance normalizer
        self.mel_normalizer = GlobalMVN(stats_mel_file)
        self.enc_postnet = Encoder_Postnet(embed_size)
        self.use_mel = (n_mels > 0)

        if self.use_mel:
            self.decoder = Decoder(dec_num_block,embed_size,n_mels,dec_nhead,dropout,device=device)
            self.postnet = module.PostNet(n_mels,output_dim,(output_dim//2*2))
        else:
            print(f"fix me")

    def forward(self,spec,mel,characters,phone,pitch,beat,pos_text=True,pos_char=None,
                pos_spec=None):
        encoder_out,text_phone=self.encoder(characters.squeeze(2),pos=pos_char)
        post_out = self.enc_postnet(encoder_out,phone,text_phone,pitch,beat)
        mel_output,att_weight = self.decoder(post_out,pos=pos_spec)
        output = self.postnet(mel_output)
        
        spec,_ = self.normalizer(spec,pos_spec)
        mel,_ = self.mel_normalizer(mel,pos_spec)
        return output,att_weight, mel_output, spec, mel


class Transformer_noGLUSVS_norm(nn.Module):
    def __init__(self,stats_file,stats_mel_file,phone_size,embed_size,hidden_size,glu_num_layers,dropout,\
                 dec_num_block,dec_nhead,output_dim,n_mels=80,local_gaussian=False,device="cuda"):
        super(Transformer_noGLUSVS_norm,self).__init__()
        self.encoder = SA_Encoder(phone_size,embed_size,hidden_size,dropout)
        self.normalizer = GlobalMVN(stats_file)   # FIX ME, add utterance normalizer
        self.mel_normalizer = GlobalMVN(stats_mel_file)
        self.enc_postnet = Encoder_Postnet(embed_size)
        self.use_mel = (n_mels > 0)

        if self.use_mel:
            self.decoder = Decoder_noGLU(dec_num_block,embed_size,n_mels,dec_nhead,dropout,device=device)
            self.postnet = module.PostNet(n_mels,output_dim,(output_dim//2*2))
        else:
            print(f"fix me")

    def forward(self,spec,mel,characters,phone,pitch,beat,pos_text=True,pos_char=None,
                pos_spec=None):
        encoder_out,text_phone=self.encoder(characters.squeeze(2),pos=pos_char)
        post_out = self.enc_postnet(encoder_out,phone,text_phone,pitch,beat)
        mel_output,att_weight = self.decoder(post_out,pos=pos_spec)
        output = self.postnet(mel_output)
        
        spec,_ = self.normalizer(spec,pos_spec)
        mel,_ = self.mel_normalizer(mel,pos_spec)
        return output,att_weight, mel_output, spec, mel





def _test():
    # debug test

    import random
    random.seed(7)
    batch_size = 16
    max_length = 500
    char_max_length = 50
    feat_dim = 1324
    phone_size = 67
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_len_list = []
    for i in range(batch_size):
        seq_len_list.append(random.randint(0, max_length))

    char_seq_len_list = []
    for i in range(batch_size):
        char_seq_len_list.append(random.randint(0, char_max_length))

    spec = torch.zeros(batch_size, max_length, feat_dim)
    phone = torch.zeros(batch_size, max_length, 1).long()
    pitch = torch.zeros(batch_size, max_length, 1).long()
    beat = torch.zeros(batch_size, max_length, 1).long()
    char = torch.zeros([batch_size, char_max_length, 1]).long()
    for i in range(batch_size):
        length = seq_len_list[i]
        char_length = char_seq_len_list[i]
        spec[i, :length, :] = torch.randn(length, feat_dim)
        phone[i, :length, :] = torch.randint(0, phone_size, (length, 1)).long()
        pitch[i, :length, :] = torch.randint(0, 200, (length, 1)).long()
        beat[i, :length, :] = torch.randint(0, 2, (length, 1)).long()
        char[i, :char_length, :] = torch.randint(0, phone_size, (char_length, 1)).long()

    seq_len = torch.from_numpy(np.array(seq_len_list)).to(device)
    char_seq_len = torch.from_numpy(np.array(char_seq_len_list)).to(device)
    spec = spec.to(device)
    phone = phone.to(device)
    pitch = pitch.to(device)
    beat = beat.to(device)
    print(seq_len.size())
    print(char_seq_len.size())
    print(spec.size())
    print(phone.size())
    print(pitch.size())
    print(beat.size())
    print(type(beat))

    hidden_size = 256
    embed_size = 256
    nhead = 4
    dropout = 0.1
    activation = 'relu'
    glu_kernel = 3
    num_dec_block = 3
    glu_num_layers = 1
    num_glu_block = 3
    
    #test encoder and encoer_postnet
    encoder = Encoder(phone_size, embed_size, hidden_size, dropout, num_glu_block, num_layers=glu_num_layers, glu_kernel=glu_kernel)
    encoder_out, text_phone = encoder(phone.squeeze(2))
    print('encoder_out.size():',encoder_out.size())
    
    post = Encoder_Postnet(embed_size)
    post_out = post(encoder_out, phone, text_phone, pitch.float(), beat)
    print('post_net_out.size():',post_out.size())
    
    
    # test model as a whole
    # model = GLU_Transformer(phone_size, hidden_size, embed_size, glu_num_layers, dropout, num_dec_block, nhead, feat_dim)
    # spec_pred = model(char, phone, pitch, beat, src_key_padding_mask=seq_len, char_key_padding_mask=char_seq_len)
    # print(spec_pred)

    # test decoder
    out_from_encoder = torch.zeros(batch_size, max_length, hidden_size)
    for i in range(batch_size):
        length = seq_len_list[i]
        out_from_encoder[i, :length, :] = torch.randn(length, hidden_size)
    decoder = Decoder(num_dec_block, embed_size, feat_dim, nhead, dropout)
    decoder_out, att = decoder(out_from_encoder, src_key_padding_mask=seq_len)
    print(decoder_out.size())
    print(att.size())



if __name__ == "__main__":
    _test()



