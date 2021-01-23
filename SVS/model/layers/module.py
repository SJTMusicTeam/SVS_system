"""Copyright [2020] [Jiatong Shi & Hailan Lin].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# !/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi, Hailan Lin)

import copy
import math
import numpy as np
from SVS.model.layers.pretrain_module import Attention
import torch
import torch.nn as nn

from torch.nn import Dropout
from torch.nn import functional as F
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module

# from torch.nn.parameter import Parameter
from torch.nn import ModuleList
from torch.nn import TransformerEncoderLayer

# from torch.nn.init import xavier_normal_
# from torch.nn.init import constant_
import torch.nn.init as init
from torch.nn.init import xavier_uniform_


SCALE_WEIGHT = 0.5 ** 0.5


class GatedConv(nn.Module):
    """GatedConv."""

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        """init."""
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0),
        )
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        """forward."""
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    """Stacked CNN class."""

    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        """init."""
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        """forward."""
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class GLU(nn.Module):
    """GLU."""

    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, input_size):
        """init."""
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, emb):
        """forward."""
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)

        emb_remap = _shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return out.squeeze(3).contiguous()


class PositionalEncoding(nn.Module):
    """Positional Encoding.

    Modified from
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, device="cuda"):
        """init."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pe = pe.to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Input of forward function.

        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CBHG(nn.Module):
    """CBHG Module."""

    def __init__(
        self,
        hidden_size,
        K=16,
        projection_size=256,
        num_gru_layers=2,
        max_pool_kernel_size=2,
        is_post=False,
    ):
        """init."""
        # :param hidden_size: dimension of hidden unit
        # :param K: # of convolution banks
        # :param projection_size: dimension of projection unit
        # :param num_gru_layers: # of layers of GRUcell
        # :param max_pool_kernel_size: max pooling kernel size
        # :param is_post: whether post processing or not
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(
            nn.Conv1d(
                in_channels=projection_size,
                out_channels=hidden_size,
                kernel_size=1,
                padding=int(np.floor(1 / 2)),
            )
        )

        for i in range(2, K + 1):
            self.convbank_list.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=i,
                    padding=int(np.floor(i / 2)),
                )
            )

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K

        self.conv_projection_1 = nn.Conv1d(
            in_channels=convbank_outdim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.conv_projection_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=projection_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)

        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(
            self.projection_size,
            self.hidden_size // 2,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

    def _conv_fit_dim(self, x, kernel_size=3):
        """_conv_fit_dim."""
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        """forward."""
        input_ = input_.contiguous()
        # batch_size = input_.size(0)
        # total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(
            zip(self.convbank_list, self.batchnorm_list)
        ):
            convbank_input = torch.relu(
                batchnorm(self._conv_fit_dim(conv(convbank_input), k + 1).contiguous())
            )
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]

        # Projection
        conv_projection = torch.relu(
            self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat)))
        )
        conv_projection = (
            self.batchnorm_proj_2(
                self._conv_fit_dim(self.conv_projection_2(conv_projection))
            )
            + input_
        )

        # Highway networks
        highway = self.highway.forward(conv_projection.transpose(1, 2))

        # Bidirectional GRU

        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(nn.Module):
    """Highway network."""

    def __init__(self, num_units, num_layers=4):
        """init."""
        # :param num_units: dimension of hidden unit
        # :param num_layers: # of highway layers

        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))

    def forward(self, input_):
        """forward."""
        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = torch.relu(fc1.forward(out))
            t_ = torch.sigmoid(fc2.forward(out))

            c = 1.0 - t_
            out = h * t_ + out * c

        return out


class Transformer_noGLULayer(Module):
    """Transformer_noGLULayer."""

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(Transformer_noGLULayer, self).__init__()
        self.self_attn = Attention(
            h=nhead, num_hidden=d_model, local_gaussian=local_gaussian
        )
        self.GLU = GLU(1, d_model, glu_kernel, dropout, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        """__setstate__."""
        if "activation" not in state:
            state["activation"] = F.relu
        super(Transformer_noGLULayer, self).__setstate__(state)

    def forward(self, src, mask=None, query_mask=None):
        """forward."""
        src1 = self.norm1(src)
        src2, att_weight = self.self_attn(src1, src1, mask=mask, query_mask=query_mask)
        src3 = src + self.dropout1(src2)
        src3 = src3 * SCALE_WEIGHT
        # src4 = self.norm2(src3)
        # src5 = self.GLU(src4)
        # src5 = src5.transpose(1, 2)
        # src6 = src3 + self.dropout2(src5)
        # src6 = src6 * SCALE_WEIGHT
        return src3, att_weight


class TransformerGLULayer(Module):
    """TransformerGLULayer."""

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(TransformerGLULayer, self).__init__()
        self.self_attn = Attention(
            h=nhead, num_hidden=d_model, local_gaussian=local_gaussian
        )
        self.GLU = GLU(1, d_model, glu_kernel, dropout, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        """__setstate__."""
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerGLULayer, self).__setstate__(state)

    def forward(self, src, mask=None, query_mask=None):
        """forward."""
        src1 = self.norm1(src)
        src2, att_weight = self.self_attn(src1, src1, mask=mask, query_mask=query_mask)
        src3 = src + self.dropout1(src2)
        src3 = src3 * SCALE_WEIGHT
        src4 = self.norm2(src3)
        src5 = self.GLU(src4)
        src5 = src5.transpose(1, 2)
        src6 = src3 + self.dropout2(src5)
        src6 = src6 * SCALE_WEIGHT
        return src6, att_weight


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the
            TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        """init."""
        super(TransformerEncoder, self).__init__()
        assert num_layers > 0
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, query_mask=None):
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask:
                the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # #type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        output = src

        for mod in self.layers:
            output, att_weight = mod(output, mask=mask, query_mask=query_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, att_weight


class Transformer(nn.Module):
    """Transformer encoder based Singing Voice synthesis.

    Args:
        hidden_state:
            the number of expected features in the encoder/decoder inputs.
        nhead: the number of heads in the multiheadattention models.
        num_block: the number of sub-encoder-layers in the encoder.
        fc_dim: the dimension of the feedforward network model.
        dropout: the dropout value.
        pos_enc: True if positional encoding is used.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_state=512,
        output_dim=128,
        nhead=4,
        num_block=6,
        fc_dim=512,
        dropout=0.1,
        pos_enc=True,
    ):
        """init."""
        super(Transformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_state)

        self.pos_encoder = PositionalEncoding(hidden_state, dropout)
        # define a single transformer encoder layer

        encoder_layer = TransformerEncoderLayer(hidden_state, nhead, fc_dim, dropout)
        self.input_norm = LayerNorm(hidden_state)
        encoder_norm = LayerNorm(hidden_state)
        self.encoder = TransformerEncoder(encoder_layer, num_block, encoder_norm)
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
        """forward."""
        src = torch.transpose(src, 0, 1)
        embed = self.input_fc(src)
        embed = self.input_norm(embed)
        if self.pos_enc:

            embed, att_weight = self.encoder(embed)
            embed = embed * math.sqrt(self.hidden_state)
            embed = self.pos_encoder(embed)
        memory, att_weight = self.encoder(
            embed, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        print(memory.size())
        output = self.postnet(memory)
        output = torch.transpose(output, 0, 1)
        return output, att_weight


class PostNet(nn.Module):
    """CBHG Network (mel --> linear)."""

    def __init__(self, input_channel, output_channel, hidden_state):
        """init."""
        super(PostNet, self).__init__()
        self.pre_projection = nn.Conv1d(
            input_channel,
            hidden_state,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        self.cbhg = CBHG(hidden_state, projection_size=hidden_state)
        self.post_projection = nn.Conv1d(
            hidden_state,
            output_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        """forward."""
        x = x.transpose(1, 2)
        x = self.pre_projection(x)
        x = self.cbhg(x).transpose(1, 2)
        output = self.post_projection(x).transpose(1, 2)

        return output


def _get_activation_fn(activation):
    """_get_activation_fn."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    """_get_clones."""
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _shape_transform(x):
    """Tranform the size of the tensors to fit for conv input."""
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class MultiHeadAttentionLayer(nn.Module):
    """MultiHeadAttentionLayer."""

    def __init__(self, hid_dim, n_heads, dropout, device):
        """init."""
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        """forward."""
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """PositionwiseFeedforwardLayer."""

    def __init__(self, hid_dim, pf_dim, dropout):
        """init."""
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """forward."""
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x
