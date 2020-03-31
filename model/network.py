import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp
import math
import glu

class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, para):
        """
        :param para: class that contains all parameters
        """
        super(Encoder, self).__init__()
        #self.alpha = nn.Parameter(t.ones(1))
        
        self.emb_phone = nn.Embedding(para['phone_size'], para['emb_dim'])
        #full connected
        self.fc_1 = nn.Linear(para['emb_dim'], para['GLU_in_dim'])
        
        self.GLU = GLU(para['num_layers'], para['hidden_size'], para['kernel_size'], para['dropout'], para['GLU_in_dim'])
        
        self.fc_2 = nn.Linear(para['GLU_out_dim'], para['emb_dim'])
        

    def forward(self, input):

        embedded_phone = self.emb_phone(input)    # [src len, batch size, emb dim]
        glu_out = self.fc_2(self.GLU(self.fc_1(embedded_phone)))
        
        out = embedded_phone + glu_out
        out = out *  math.sqrt(0.5)

        return out


class Decoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self):
        super(MelDecoder, self).__init__()

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
