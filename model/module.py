#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi, Hailan Lin)

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.init import xavier_normal_
from torch.nn.init import constant_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import numpy as np
import math
import copy

SCALE_WEIGHT = 0.5 ** 0.5


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_size, out_channels=2 * input_size,
                                     kernel_size=(width, 1), stride=(1, 1),
                                     padding=(width // 2 * (1 - nopad), 0))
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    """ Stacked CNN class """

    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class GLU(nn.Module):
    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, input_size):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width, dropout)

    def forward(self, emb):
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)

        emb_remap = _shape_transform(emb_remap)
        out = self.cnn(emb_remap)
        
        return out.squeeze(3).contiguous()


class PositionalEncoding(nn.Module):
    """ Positional Encoding.
    Modified from
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CBHG(nn.Module):
    """
    CBHG Module
    """
    def __init__(self, hidden_size, K=16, projection_size = 256, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        
        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=True)


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):

        input_ = input_.contiguous()
        batch_size = input_.size(0)
        total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = torch.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = torch.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # Highway networks
        highway = self.highway.forward(conv_projection.transpose(1,2))
        

        # Bidirectional GRU
        
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(nn.Module):
    """
    Highway network
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = torch.relu(fc1.forward(out))
            t_ = torch.sigmoid(fc2.forward(out))

            c = 1. - t_
            out = h * t_ + out * c

        return out


class MultiheadAttention(nn.Module):
    """
    Multiheaded Scaled Dot Product Attention
    Implements equation:
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Similarly to the above, d_k = d_v = d_model / h
    Inputs
      init:
        nheads : integer # of attention heads
        d_model : model dimensionality
        d_head : dimensionality of a single head
      forward:
        query : [batch size, sequence length, d_model]
        key: [batch size, sequence length, d_model]
        value: [batch size, sequence length, d_model]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths
    Output
      result : [batch_size, sequence length, d_model]
    """

    def __init__(self, nheads, d_model):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert d_model % nheads == 0
        self.d_head = d_model // nheads
        self.nheads = nheads
        self.Q_fc = nn.Linear(d_model, d_model, bias=False)
        self.K_fc = nn.Linear(d_model, d_model, bias=False)
        self.V_fc = nn.Linear(d_model, d_model, bias=False)
        self.output_fc = nn.Linear(d_model, d_model, bias=False)
        self.attn = None

    def forward(self, query, key, value, unseen_mask=False, src_lengths=None):
        # 1. Fully-connected layer on q, k, v then
        # 2. Split heads on q, k, v
        # (batch_size, seq_len, d_model) -->
        # (batch_size, nheads, seq_len, d_head)
        query = split_heads(self.Q_fc(query), self.nheads)
        key = split_heads(self.K_fc(key), self.nheads)
        value = split_heads(self.V_fc(value), self.nheads)

        # 4. Scaled dot product attention
        # (batch_size, nheads, seq_len, d_head)
        x, self.attn = scaled_dot_prod_attn(
            query=query,
            key=key,
            value=value,
            unseen_mask=unseen_mask,
            src_lengths=src_lengths,
        )

        # 5. Combine heads
        x = combine_heads(x)

        # 6. Fully-connected layer for output
        return self.output_fc(x)


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        src2, att_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, att_weight


class TransformerGLULayer(Module):
    """TransformerGLULayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        glu_kernel: the kernel size of glu block

    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu",
        glu_kernel=3):
        super(TransformerGLULayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.GLU = GLU(1, d_model, glu_kernel, dropout, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, att_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.GLU(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, att_weight


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output, att_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, att_weight


class Transformer(nn.Module):
    """Transformer encoder based Singing Voice synthesis
    Args:
        hidden_state: the number of expected features in the encoder/decoder inputs.
        nhead: the number of heads in the multiheadattention models.
        num_block: the number of sub-encoder-layers in the encoder.
        fc_dim: the dimension of the feedforward network model.
        dropout: the dropout value.
        pos_enc: True if positional encoding is used.
    """
    def __init__(self, input_dim=128, hidden_state=512, output_dim=128, nhead=4,
        num_block=6, fc_dim=512, dropout=0.1, pos_enc=True):
        super(Transformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_state)

        self.pos_encoder = PositionalEncoding(hidden_state, dropout)
        # define a single transformer encoder layer

        encoder_layer = TransformerEncoderLayer(hidden_state, nhead, fc_dim, dropout)
        self.input_norm = LayerNorm(hidden_state)
        encoder_norm = LayerNorm(hidden_state)
        self.encoder = TransformerEncoder(encoder_layer,
            num_block, encoder_norm)
        self.postnet = PostNet(hidden_state, output_dim, hidden_state)
        self.hidden_state = hidden_state
        self.pos_enc = pos_enc

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = torch.transpose(src, 0, 1)
        embed = self.input_fc(src)
        embed = self.input_norm(embed)
        if self.pos_enc:

            embed, att_weight = self.encoder(embed)
            embed = embed * math.sqrt(self.hidden_state)
            embed = self.pos_encoder(embed)
        memory, att_weight = self.encoder(
            embed,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask)
        print(memory.size())
        output = self.postnet(memory)
        output = torch.transpose(output, 0, 1)
        return output, att_weight


class PostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self, input_channel, output_channel, hidden_state):
        super(PostNet, self).__init__()
        self.pre_projection = nn.Conv1d(input_channel, hidden_state, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=True)
        self.cbhg = CBHG(hidden_state, projection_size=hidden_state)
        self.post_projection = nn.Conv1d(hidden_state, output_channel, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pre_projection(x)
        x = self.cbhg(x).transpose(1, 2)
        output = self.post_projection(x).transpose(1, 2)

        return output


def create_src_lengths_mask(batch_size, src_lengths):
    max_srclen = src_lengths.max()
    src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def apply_masks(scores, batch_size, unseen_mask, src_lengths):
    seq_len = scores.shape[-1]

    # [1, seq_len, seq_len]
    sequence_mask = torch.ones(seq_len, seq_len).unsqueeze(0).int()
    if unseen_mask:
        # [1, seq_len, seq_len]
        sequence_mask = (
            torch.tril(torch.ones(seq_len, seq_len), diagonal=0).unsqueeze(0).int()
        )

    if src_lengths is not None:
        # [batch_size, 1, seq_len]
        src_lengths_mask = create_src_lengths_mask(
            batch_size=batch_size, src_lengths=src_lengths
        ).unsqueeze(-2)

        # [batch_size, seq_len, seq_len]
        sequence_mask = sequence_mask & src_lengths_mask

    # [batch_size, 1, seq_len, seq_len]
    sequence_mask = sequence_mask.unsqueeze(1)

    scores = scores.masked_fill(sequence_mask == 0, -np.inf)
    return scores


def scaled_dot_prod_attn(query, key, value, unseen_mask=False, src_lengths=None):
    """
    Scaled Dot Product Attention
    Implements equation:
    Attention(Q, K, V) = softmax(QK^T/\sqrt{d_k})V
    Inputs:
      query : [batch size, nheads, sequence length, d_k]
      key : [batch size, nheads, sequence length, d_k]
      value : [batch size, nheads, sequence length, d_v]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths
    Outputs:
      attn: [batch size, sequence length, d_v]
    Note that in this implementation d_q = d_k = d_v = dim
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_k)
    if unseen_mask or src_lengths is not None:
        scores = apply_masks(
            scores=scores,
            batch_size=query.shape[0],
            unseen_mask=unseen_mask,
            src_lengths=src_lengths,
        )
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def split_heads(X, nheads):
    """
    Split heads:
    1) Split (reshape) last dimension (size d_model) into nheads, d_head
    2) Transpose X from (batch size, sequence length, nheads, d_head) to
        (batch size, nheads, sequence length, d_head)
    Inputs:
      X : [batch size, sequence length, nheads * d_head]
      nheads : integer
    Outputs:
      [batch size,  nheads, sequence length, d_head]
    """
    last_dim = X.shape[-1]
    assert last_dim % nheads == 0
    X_last_dim_split = X.view(list(X.shape[:-1]) + [nheads, last_dim // nheads])
    return X_last_dim_split.transpose(1, 2)


def combine_heads(X):
    """
    Combine heads (the inverse of split heads):
    1) Transpose X from (batch size, nheads, sequence length, d_head) to
        (batch size, sequence length, nheads, d_head)
    2) Combine (reshape) last 2 dimensions (nheads, d_head) into 1 (d_model)
    Inputs:
      X : [batch size * nheads, sequence length, d_head]
      nheads : integer
      d_head : integer
    Outputs:
      [batch_size, seq_len, d_model]
    """
    X = X.transpose(1, 2)
    nheads, d_head = X.shape[-2:]
    return X.contiguous().view(list(X.shape[:-2]) + [nheads * d_head])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)