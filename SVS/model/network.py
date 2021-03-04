"""Copyright [2020] [Jiatong Shi & Shuai Guo & Hailan Lin].

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

# debug only
# import sys
# sys.path.append("/Users/jiatongshi/projects/svs_system/SVS_system")

import math
import numpy as np
from pathlib import Path
from SVS.model.layers.conformer_related import Conformer_block
import SVS.model.layers.module as module
from SVS.model.layers.pretrain_module import Attention
from SVS.model.layers.pretrain_module import clones
from SVS.model.layers.pretrain_module import FFN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from typing import Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
    """
    return ~make_pad_mask(lengths, xs, length_dim)


class Encoder(nn.Module):
    """Encoder Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        hidden_size,
        dropout,
        GLU_num,
        num_layers=1,
        glu_kernel=3,
    ):
        """init."""
        # :param para: dictionary that contains all parameters
        super(Encoder, self).__init__()

        self.emb_phone = nn.Embedding(phone_size, embed_size)
        # full connected
        self.fc_1 = nn.Linear(embed_size, hidden_size)

        self.GLU_list = nn.ModuleList()
        for i in range(int(GLU_num)):
            self.GLU_list.append(
                module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)
            )
        # self.GLU =
        # module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)

        self.fc_2 = nn.Linear(hidden_size, embed_size)

    def forward(self, text_phone, pos=None):
        """forward."""
        # text_phone dim: [batch_size, text_phone_length]
        # output dim : [batch_size, text_phone_length, embedded_dim]

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
    """SA_Encoder."""

    def __init__(
        self, phone_size, embed_size, hidden_size, dropout, num_blocks=3, nheads=4
    ):
        """init."""
        super(SA_Encoder, self).__init__()
        self.layers = clones(Attention(hidden_size), int(num_blocks))
        self.ffns = clones(FFN(hidden_size), int(num_blocks))
        self.emb_phone = nn.Embedding(phone_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, hidden_size)

    def forward(self, text_phone, pos):
        """forward."""
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
        return x, text_phone


class Conformer_Encoder(nn.Module):
    """Conformer_Encoder Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        cnn_module_kernel=31,
        padding_idx=-1,
    ):
        """__init__."""
        super(Conformer_Encoder, self).__init__()
        self.emb_phone = nn.Embedding(phone_size, embed_size)
        self.conformer_block = Conformer_block(
            embed_size,
            attention_dim,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            normalize_before,
            concat_after,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            macaron_style,
            pos_enc_layer_type,
            selfattention_layer_type,
            activation_type,
            use_cnn_module,
            cnn_module_kernel,
            padding_idx,
        )

    def forward(self, text_phone, pos, length):
        """forward."""
        if self.training:
            # query_mask = pos.ne(0).type(torch.float)
            pass
        else:
            # query_mask = None
            pass
        mask = pos.eq(0).unsqueeze(1).repeat(1, text_phone.size(1), 1)

        embedded_phone = self.emb_phone(text_phone)

        # print(text_phone)
        # print(mask)

        # print(f"embedded_phone: {np.shape(embedded_phone)}")

        # length = torch.max(length,dim=1)[0]
        # length = [batch size]
        # embedded_phone = embedded_phone[:, : max(length)]
        # for data parallel
        # src_mask = make_non_pad_mask(length.tolist())
        # .to(embedded_phone.device).unsqueeze(-2)

        # print(f"length: {length}, text_phone: {np.shape(text_phone)},
        # src_mask: {np.shape(mask)}, embedded_phone:
        # {np.shape(embedded_phone)}")

        x, masks = self.conformer_block(embedded_phone, mask)

        return x, text_phone


class Encoder_Postnet(nn.Module):
    """Encoder Postnet."""

    def __init__(self, embed_size):
        """init."""
        super(Encoder_Postnet, self).__init__()

        self.fc_pitch = nn.Linear(1, embed_size)
        # Remember! embed_size must be even!!
        self.fc_pos = nn.Linear(embed_size, embed_size)
        # only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, embed_size)
        self.pos = module.PositionalEncoding(embed_size)

    def aligner(self, encoder_out, align_phone, text_phone):
        """aligner."""
        # align_phone = [batch_size, align_phone_length]
        # text_phone = [batch_size, text_phone_length]
        # align_phone_length( = frame_num) > text_phone_length
        # batch
        align_phone = align_phone.long()
        for i in range(align_phone.shape[0]):
            before_text_phone = text_phone[i][0]
            encoder_ind = 0
            line = encoder_out[i][0].unsqueeze(0)
            # frame
            for j in range(1, align_phone.shape[1]):
                if align_phone[i][j] == before_text_phone:
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
                else:
                    encoder_ind += 1
                    if encoder_ind >= text_phone[i].size()[0]:
                        break
                    before_text_phone = text_phone[i][encoder_ind]
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
            if i == 0:
                out = line.unsqueeze(0)
            else:
                out = torch.cat((out, line.unsqueeze(0)), dim=0)

        return out

    def forward(self, encoder_out, align_phone, text_phone, pitch, beats):
        """pitch/beats:[batch_size, frame_num]->[batch_size, frame_num，1]."""
        # batch_size = pitch.shape[0]
        # frame_num = pitch.shape[1]
        # embedded_dim = encoder_out.shape[2]

        aligner_out = self.aligner(encoder_out, align_phone, text_phone)

        pitch = self.fc_pitch(pitch)
        out = aligner_out + pitch

        beats = self.emb_beats(beats.squeeze(2))
        out = out + beats

        pos_encode = self.pos(torch.transpose(aligner_out, 0, 1))
        pos_out = self.fc_pos(torch.transpose(pos_encode, 0, 1))

        out = out + pos_out

        return out


class Encoder_Postnet_combine(nn.Module):
    """Encoder Postnet."""

    def __init__(self, embed_size, singer_size):
        """init."""
        super(Encoder_Postnet_combine, self).__init__()

        self.fc_pitch = nn.Linear(1, embed_size)
        # Remember! embed_size must be even!!
        self.fc_pos = nn.Linear(embed_size, embed_size)
        # only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, embed_size)
        self.pos = module.PositionalEncoding(embed_size)

        self.emb_singer = nn.Embedding(singer_size, embed_size)
        self.fc_out = nn.Linear(2 * embed_size, embed_size)

    def aligner(self, encoder_out, align_phone, text_phone):
        """aligner."""
        # align_phone = [batch_size, align_phone_length]
        # text_phone = [batch_size, text_phone_length]
        # align_phone_length( = frame_num) > text_phone_length
        # batch
        align_phone = align_phone.long()
        for i in range(align_phone.shape[0]):
            before_text_phone = text_phone[i][0]
            encoder_ind = 0
            line = encoder_out[i][0].unsqueeze(0)
            # frame
            for j in range(1, align_phone.shape[1]):
                if align_phone[i][j] == before_text_phone:
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
                else:
                    encoder_ind += 1
                    if encoder_ind >= text_phone[i].size()[0]:
                        break
                    before_text_phone = text_phone[i][encoder_ind]
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
            if i == 0:
                out = line.unsqueeze(0)
            else:
                out = torch.cat((out, line.unsqueeze(0)), dim=0)

        return out

    def forward(self, encoder_out, align_phone, text_phone, pitch, beats, singer_vec):
        """pitch/beats:[batch_size, frame_num]->[batch_size, frame_num，1]."""
        # singer_vec=[        # repeat singer_id for length times
        #     [1,1,...,1]
        #     [0,0,...,0]
        #     [5,5,...,5]
        # ]
        # batch_size = pitch.shape[0]
        # frame_num = pitch.shape[1]
        # embedded_dim = encoder_out.shape[2]

        aligner_out = self.aligner(encoder_out, align_phone, text_phone)

        pitch = self.fc_pitch(pitch)
        out = aligner_out + pitch

        beats = self.emb_beats(beats.squeeze(2))
        out = out + beats

        pos_encode = self.pos(torch.transpose(aligner_out, 0, 1))
        pos_out = self.fc_pos(torch.transpose(pos_encode, 0, 1))
        out = out + pos_out

        singer_embed = self.emb_singer(
            singer_vec.squeeze(-1)
        )  # [batch size, length, emb_size]
        out = torch.cat(
            (out, singer_embed), dim=2
        )  # [batch size, length, 2 * emb_size]

        out = F.leaky_relu(self.fc_out(out))

        return out


class Decoder_noGLU(nn.Module):
    """Decoder Network."""

    def __init__(
        self,
        num_block,
        hidden_size,
        output_dim,
        nhead=4,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(Decoder_noGLU, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.Transformer_noGLULayer(
            hidden_size,
            nhead,
            dropout,
            activation,
            glu_kernel,
            local_gaussian=local_gaussian,
            device=device,
        )
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size = hidden_size

    def forward(self, src, pos):
        """forward."""
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
    """Decoder Network."""

    def __init__(
        self,
        num_block,
        hidden_size,
        output_dim,
        nhead=4,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(Decoder, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.TransformerGLULayer(
            hidden_size,
            nhead,
            dropout,
            activation,
            glu_kernel,
            local_gaussian=local_gaussian,
            device=device,
        )
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size = hidden_size

    def forward(self, src, pos):
        """forward."""
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        src = self.input_norm(src)
        memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output, att_weight


class Conformer_Decoder(nn.Module):
    """Conformer_Decoder Network."""

    def __init__(
        self,
        embed_size,
        n_mels,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        cnn_module_kernel=31,
        padding_idx=-1,
    ):
        """init."""
        super(Conformer_Decoder, self).__init__()

        self.conformer_block = Conformer_block(
            embed_size,
            attention_dim,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            normalize_before,
            concat_after,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            macaron_style,
            pos_enc_layer_type,
            selfattention_layer_type,
            activation_type,
            use_cnn_module,
            cnn_module_kernel,
            padding_idx,
        )
        self.output_fc = nn.Linear(embed_size, n_mels)

    def forward(self, src, pos):
        """forward."""
        if self.training:
            pass
            # query_mask = pos.ne(0).type(torch.float)
        else:
            pass
            # query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        x, masks = self.conformer_block(src, mask)
        output = self.output_fc(x)

        return output


class GLU_TransformerSVS(nn.Module):
    """Transformer Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        hidden_size,
        glu_num_layers,
        dropout,
        dec_num_block,
        dec_nhead,
        output_dim,
        n_mels=-1,
        double_mel_loss=True,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(GLU_TransformerSVS, self).__init__()
        self.encoder = Encoder(
            phone_size,
            embed_size,
            hidden_size,
            dropout,
            glu_num_layers,
            num_layers=1,
            glu_kernel=3,
        )
        self.enc_postnet = Encoder_Postnet(embed_size)

        self.use_mel = n_mels > 0
        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False

        if self.use_mel:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                n_mels,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                output_dim,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        output = self.postnet(mel_output2)

        return output, att_weight, mel_output, mel_output2


class GLU_TransformerSVS_combine(nn.Module):
    """Transformer Network."""

    def __init__(
        self,
        phone_size,
        singer_size,
        embed_size,
        hidden_size,
        glu_num_layers,
        dropout,
        dec_num_block,
        dec_nhead,
        output_dim,
        n_mels=-1,
        double_mel_loss=True,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(GLU_TransformerSVS_combine, self).__init__()
        self.encoder = Encoder(
            phone_size,
            embed_size,
            hidden_size,
            dropout,
            glu_num_layers,
            num_layers=1,
            glu_kernel=3,
        )
        self.enc_postnet = Encoder_Postnet_combine(embed_size, singer_size)

        self.use_mel = n_mels > 0
        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False

        if self.use_mel:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                n_mels,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                output_dim,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        singer_vec,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(
            encoder_out, phone, text_phone, pitch, beat, singer_vec
        )
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        output = self.postnet(mel_output2)

        return output, att_weight, mel_output, mel_output2


class LSTMSVS(nn.Module):
    """LSTM singing voice synthesis model."""

    def __init__(
        self,
        embed_size=512,
        d_model=512,
        d_output=1324,
        num_layers=2,
        phone_size=87,
        n_mels=-1,
        double_mel_loss=True,
        dropout=0.1,
        device="cuda",
        use_asr_post=False,
    ):
        """init."""
        super(LSTMSVS, self).__init__()

        if use_asr_post:
            self.input_fc = nn.Linear(phone_size - 1, d_model)
        else:
            self.input_embed = nn.Embedding(phone_size, embed_size)

        self.linear_wrapper = nn.Linear(embed_size, d_model)

        self.phone_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.linear_wrapper2 = nn.Linear(d_model * 2, d_model)

        self.pos = module.PositionalEncoding(d_model)
        # Remember! embed_size must be even!!
        assert embed_size % 2 == 0
        self.fc_pos = nn.Linear(d_model, d_model)

        self.pos_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc_pitch = nn.Linear(1, d_model)
        self.linear_wrapper3 = nn.Linear(d_model * 2, d_model)
        self.pitch_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, d_model)
        self.linear_wrapper4 = nn.Linear(d_model * 2, d_model)
        self.beats_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.output_fc = nn.Linear(d_model * 2, d_output)

        self.use_mel = n_mels > 0
        self.n_mels = n_mels
        if self.use_mel:
            self.output_mel = nn.Linear(d_model * 2, n_mels)
            self.postnet = module.PostNet(n_mels, d_output, (d_output // 2 * 2))

            self.double_mel_loss = double_mel_loss
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
        else:
            self.output_fc = nn.Linear(d_model * 2, d_output)

            self.double_mel_loss = False

        self._reset_parameters()

        self.use_asr_post = use_asr_post
        self.d_model = d_model

    def forward(self, phone, pitch, beats):
        """forward."""
        if self.use_asr_post:
            out = self.input_fc(phone.squeeze(-1))
        else:
            out = self.input_embed(phone.squeeze(-1))
        out = F.leaky_relu(out)
        out = F.leaky_relu(self.linear_wrapper(out))
        out, _ = self.phone_lstm(out)
        out = F.leaky_relu(self.linear_wrapper2(out))

        pos = self.pos(out)
        # pos_encode = self.fc_pos(pos)
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
            mel_output = self.output_mel(out)
            if self.double_mel_loss:
                mel_output2 = self.double_mel(mel_output)
            else:
                mel_output2 = mel_output
            output = self.postnet(mel_output2)
            # out = self.postnet(mel)
            return output, (h0, c0), mel_output, mel_output2
        else:
            out = self.output_fc(out)
            return out, (h0, c0), None, None

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class LSTMSVS_combine(nn.Module):
    """LSTM singing voice synthesis model."""

    def __init__(
        self,
        embed_size=512,
        d_model=512,
        d_output=1324,
        num_layers=2,
        phone_size=87,
        singer_size=10,
        n_mels=-1,
        double_mel_loss=True,
        dropout=0.1,
        device="cuda",
        use_asr_post=False,
    ):
        """init."""
        super(LSTMSVS_combine, self).__init__()

        if use_asr_post:
            self.input_fc = nn.Linear(phone_size - 1, d_model)
        else:
            self.input_embed = nn.Embedding(phone_size, embed_size)

        self.emb_singer = nn.Embedding(singer_size, embed_size)
        self.linear_wrapper = nn.Linear(2 * embed_size, d_model)

        self.phone_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.linear_wrapper2 = nn.Linear(d_model * 2, d_model)

        self.pos = module.PositionalEncoding(d_model)
        # Remember! embed_size must be even!!
        assert embed_size % 2 == 0
        self.fc_pos = nn.Linear(d_model, d_model)

        self.pos_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc_pitch = nn.Linear(1, d_model)
        self.linear_wrapper3 = nn.Linear(d_model * 2, d_model)
        self.pitch_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, d_model)
        self.linear_wrapper4 = nn.Linear(d_model * 2, d_model)
        self.beats_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.output_fc = nn.Linear(d_model * 2, d_output)

        self.use_mel = n_mels > 0
        self.n_mels = n_mels
        if self.use_mel:
            self.output_mel = nn.Linear(d_model * 2, n_mels)
            self.postnet = module.PostNet(n_mels, d_output, (d_output // 2 * 2))

            self.double_mel_loss = double_mel_loss
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
        else:
            self.output_fc = nn.Linear(d_model * 2, d_output)

            self.double_mel_loss = False

        self._reset_parameters()

        self.use_asr_post = use_asr_post
        self.d_model = d_model

    def forward(self, phone, pitch, beats, singer_vec):
        """forward."""
        if self.use_asr_post:
            out = self.input_fc(phone.squeeze(-1))
        else:
            out = self.input_embed(phone.squeeze(-1))

        singer_embed = self.emb_singer(
            singer_vec.squeeze(-1)
        )  # [batch size, length, emb_size]
        out = torch.cat(
            (out, singer_embed), dim=2
        )  # # [batch size, length, 2 * emb_size]

        out = F.leaky_relu(out)
        out = F.leaky_relu(self.linear_wrapper(out))
        out, _ = self.phone_lstm(out)
        out = F.leaky_relu(self.linear_wrapper2(out))

        pos = self.pos(out)
        # pos_encode = self.fc_pos(pos)
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
            mel_output = self.output_mel(out)
            if self.double_mel_loss:
                mel_output2 = self.double_mel(mel_output)
            else:
                mel_output2 = mel_output
            output = self.postnet(mel_output2)
            # out = self.postnet(mel)
            return output, (h0, c0), mel_output, mel_output2
        else:
            out = self.output_fc(out)
            return out, (h0, c0), None, None

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class GRUSVS_gs(nn.Module):
    """GRU singing voice synthesis model by Guo Shuai (RUC)."""

    def __init__(
        self,
        embed_size=512,
        d_model=512,
        d_output=1324,
        num_layers=2,
        phone_size=87,
        n_mels=-1,
        dropout=0.1,
        device="cuda",
        use_asr_post=False,
    ):
        """init."""
        super(GRUSVS_gs, self).__init__()

        # Encoder
        self.embedding_phone = nn.Embedding(phone_size, embed_size)
        self.rnnEncoder = nn.GRU(embed_size + 2, d_model, bidirectional=True)
        self.fcEncoder = nn.Linear(d_model * 2, d_model)
        self.dropoutEncoder = nn.Dropout(dropout)

        # Attention
        self.attn = nn.Linear((d_model * 2) + d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)

        # Decoder
        self.rnnDecoder = nn.GRU((d_model * 2) + d_model * 2, d_model)
        self.fc_hid1 = nn.Linear((d_model * 2) + d_model * 2 + d_model, d_model * 2)
        # self.fc_hid2 = nn.Linear(2048, 1600)

        self.dropoutDecoder = nn.Dropout(dropout)

        self.use_mel = n_mels > 0
        self.n_mels = n_mels
        if self.use_mel:
            self.output_mel = nn.Linear(d_model * 2, n_mels)
            self.postnet = module.PostNet(n_mels, d_output, (d_output // 2 * 2))
        else:
            self.fc_out = nn.Linear(d_model * 2, d_output)

        self._reset_parameters()
        self.d_model = d_model
        self.d_output = d_output

    def forward(self, target, phone, pitch, beats, length, args):
        """forward."""
        # phone, pitch, beats = [batch size, len, 1]
        # target = [batch size, max_len, feat_dim]

        batch_size = np.shape(phone)[0]
        length = torch.max(length, dim=1)[0]  # length = [batch size]
        max_length = args.num_frames

        sorted_length, sorted_index = torch.sort(length, descending=True)
        phone = phone.index_select(0, sorted_index).permute(1, 0, 2)
        pitch = pitch.index_select(0, sorted_index).permute(1, 0, 2)
        beats = beats.index_select(0, sorted_index).permute(1, 0, 2)
        target = target.index_select(0, sorted_index).permute(1, 0, 2)

        # print("target", np.shape(target))

        # Encoder
        embedded_phone = self.dropoutEncoder(
            self.embedding_phone(phone[:, :, 0])
        )  # # [src len, batch size, emb dim]
        embedded = torch.cat(
            (embedded_phone.float(), beats.float(), pitch.float()), dim=2
        )  # [src len, batch size, emb_dim+2]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_length)
        packed_outputs, hidden = self.rnnEncoder(packed_embedded)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # outputs = [batch size, src len, enc hid dim * 2]
        hidden = torch.tanh(
            self.fcEncoder(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )  # hidden = [batch size, dec hid dim]

        # Decoder
        hidden = hidden.unsqueeze(0)
        outputs = torch.zeros(max_length, batch_size, self.d_model * 2).to(device)
        input_ = outputs[0]
        for t in range(0, max_length):
            input_ = input_.unsqueeze(0)  # input = [1, batch size, d_model * 2]

            # Attention
            src_len = encoder_outputs.shape[1]
            hiddenAttention = (
                hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
            )  # hidden = [batch size, src_len, dec hid dim]
            energy = torch.tanh(
                self.attn(torch.cat((hiddenAttention, encoder_outputs), dim=2))
            )  # energy = [batch size, src len, dec hid dim]
            attention = self.v(energy).squeeze(2)  # attention = [batch size, src len]
            attention_weights = F.softmax(attention, dim=1).unsqueeze(
                1
            )  # a = [batch size, 1, src len]

            weighted = torch.bmm(
                attention_weights, encoder_outputs
            )  # weighted = [batch size, 1, enc hid dim * 2]
            weighted = weighted.permute(
                1, 0, 2
            )  # weighted = [1, batch size, enc hid dim * 2]
            rnn_input = torch.cat((input_, weighted), dim=2)
            # rnn_input = [1, batch size,(enc hid dim * 2)+(enc hid dim * 2)]

            output, hidden = self.rnnDecoder(rnn_input, hidden)

            # assert (output == hidden).all()

            input_ = input_.squeeze(0)  # input = [batch size, d_model * 2]
            output = output.squeeze(0)  # output = [batch size, d_model]
            weighted = weighted.squeeze(0)  # weighted = [batch size, d_model * 2]

            prediction = self.dropoutDecoder(
                F.relu(self.fc_hid1(torch.cat((output, weighted, input_), dim=1)))
            )  # prediction = [batch size, output dim * 2]
            input_ = prediction

            # check output within the len of each_sample, pad 0 as over its len
            for each_sample in range(batch_size):
                if t < sorted_length[each_sample]:
                    outputs[t][each_sample] = prediction[each_sample]
        outputs = outputs.permute(1, 0, 2)
        if self.use_mel:
            mel = self.output_mel(outputs)
            outputs = self.postnet(mel)
            return outputs, None, mel
        else:
            outputs = self.fc_out(outputs)
            return outputs, None, None

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerSVS(nn.Module):
    """TransformerSVS."""

    def __init__(
        self,
        stats_file,
        stats_mel_file,
        phone_size,
        embed_size,
        hidden_size,
        glu_num_layers,
        dropout,
        dec_num_block,
        dec_nhead,
        output_dim,
        n_mels=80,
        double_mel_loss=True,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(TransformerSVS, self).__init__()
        self.encoder = SA_Encoder(phone_size, embed_size, hidden_size, dropout)
        self.enc_postnet = Encoder_Postnet(embed_size)
        self.use_mel = n_mels > 0

        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False

        if self.use_mel:
            self.decoder = Decoder_noGLU(
                dec_num_block,
                embed_size,
                n_mels,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder_noGLU(
                dec_num_block,
                embed_size,
                output_dim,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        spec,
        mel,
        characters,
        phone,
        pitch,
        beat,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        output = self.postnet(mel_output2)

        return output, att_weight, mel_output, mel_output2


class ConformerSVS(nn.Module):
    """Conformer Transformer Network.(Conformer encoder + plain decoder)."""

    def __init__(
        self,
        phone_size,
        embed_size,
        dec_num_block,
        dec_nhead,
        output_dim,
        n_mels=-1,
        double_mel_loss=True,
        local_gaussian=False,
        dec_dropout=0.1,
        enc_attention_dim=256,
        enc_attention_heads=4,
        enc_linear_units=2048,
        enc_num_blocks=6,
        enc_dropout_rate=0.1,
        enc_positional_dropout_rate=0.1,
        enc_attention_dropout_rate=0.0,
        enc_input_layer="conv2d",
        enc_normalize_before=True,
        enc_concat_after=False,
        enc_positionwise_layer_type="linear",
        enc_positionwise_conv_kernel_size=1,
        enc_macaron_style=False,
        enc_pos_enc_layer_type="abs_pos",
        enc_selfattention_layer_type="selfattn",
        enc_activation_type="swish",
        enc_use_cnn_module=False,
        enc_cnn_module_kernel=31,
        enc_padding_idx=-1,
        device="cuda",
    ):
        """init."""
        super(ConformerSVS, self).__init__()
        self.encoder = Conformer_Encoder(
            phone_size,
            embed_size,
            attention_dim=enc_attention_dim,
            attention_heads=enc_attention_heads,
            linear_units=enc_linear_units,
            num_blocks=enc_num_blocks,
            dropout_rate=enc_dropout_rate,
            positional_dropout_rate=enc_positional_dropout_rate,
            attention_dropout_rate=enc_attention_dropout_rate,
            input_layer=enc_input_layer,
            normalize_before=enc_normalize_before,
            concat_after=enc_concat_after,
            positionwise_layer_type=enc_positionwise_layer_type,
            positionwise_conv_kernel_size=enc_positionwise_conv_kernel_size,
            macaron_style=enc_macaron_style,
            pos_enc_layer_type=enc_pos_enc_layer_type,
            selfattention_layer_type=enc_selfattention_layer_type,
            activation_type=enc_activation_type,
            use_cnn_module=enc_use_cnn_module,
            cnn_module_kernel=enc_cnn_module_kernel,
            padding_idx=enc_padding_idx,
        )
        self.enc_postnet = Encoder_Postnet(embed_size)

        self.use_mel = n_mels > 0
        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False

        if self.use_mel:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                n_mels,
                dec_nhead,
                dec_dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            if self.double_mel_loss:
                self.double_mel = module.PostNet(n_mels, n_mels, n_mels)
            self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                output_dim,
                dec_nhead,
                dec_dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            self.postnet = module.PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(
            characters.squeeze(2), pos=pos_char, length=pos_spec
        )
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        output = self.postnet(mel_output2)

        return output, att_weight, mel_output, mel_output2


class ConformerSVS_FULL(nn.Module):
    """Conformer Transformer Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        output_dim,
        n_mels=-1,
        enc_attention_dim=256,
        enc_attention_heads=4,
        enc_linear_units=2048,
        enc_num_blocks=6,
        enc_dropout_rate=0.1,
        enc_positional_dropout_rate=0.1,
        enc_attention_dropout_rate=0.0,
        enc_input_layer="conv2d",
        enc_normalize_before=True,
        enc_concat_after=False,
        enc_positionwise_layer_type="linear",
        enc_positionwise_conv_kernel_size=1,
        enc_macaron_style=False,
        enc_pos_enc_layer_type="abs_pos",
        enc_selfattention_layer_type="selfattn",
        enc_activation_type="swish",
        enc_use_cnn_module=False,
        enc_cnn_module_kernel=31,
        enc_padding_idx=-1,
        dec_attention_dim=256,
        dec_attention_heads=4,
        dec_linear_units=2048,
        dec_num_blocks=6,
        dec_dropout_rate=0.1,
        dec_positional_dropout_rate=0.1,
        dec_attention_dropout_rate=0.0,
        dec_input_layer="conv2d",
        dec_normalize_before=True,
        dec_concat_after=False,
        dec_positionwise_layer_type="linear",
        dec_positionwise_conv_kernel_size=1,
        dec_macaron_style=False,
        dec_pos_enc_layer_type="abs_pos",
        dec_selfattention_layer_type="selfattn",
        dec_activation_type="swish",
        dec_use_cnn_module=False,
        dec_cnn_module_kernel=31,
        dec_padding_idx=-1,
        device="cuda",
    ):
        """init."""
        super(ConformerSVS_FULL, self).__init__()
        self.encoder = Conformer_Encoder(
            phone_size,
            embed_size,
            attention_dim=enc_attention_dim,
            attention_heads=enc_attention_heads,
            linear_units=enc_linear_units,
            num_blocks=enc_num_blocks,
            dropout_rate=enc_dropout_rate,
            positional_dropout_rate=enc_positional_dropout_rate,
            attention_dropout_rate=enc_attention_dropout_rate,
            input_layer=enc_input_layer,
            normalize_before=enc_normalize_before,
            concat_after=enc_concat_after,
            positionwise_layer_type=enc_positionwise_layer_type,
            positionwise_conv_kernel_size=enc_positionwise_conv_kernel_size,
            macaron_style=enc_macaron_style,
            pos_enc_layer_type=enc_pos_enc_layer_type,
            selfattention_layer_type=enc_selfattention_layer_type,
            activation_type=enc_activation_type,
            use_cnn_module=enc_use_cnn_module,
            cnn_module_kernel=enc_cnn_module_kernel,
            padding_idx=enc_padding_idx,
        )
        self.enc_postnet = Encoder_Postnet(embed_size)

        self.decoder = Conformer_Decoder(
            embed_size,
            n_mels=n_mels,
            attention_dim=dec_attention_dim,
            attention_heads=dec_attention_heads,
            linear_units=dec_linear_units,
            num_blocks=dec_num_blocks,
            dropout_rate=dec_dropout_rate,
            positional_dropout_rate=dec_positional_dropout_rate,
            attention_dropout_rate=dec_attention_dropout_rate,
            input_layer=dec_input_layer,
            normalize_before=dec_normalize_before,
            concat_after=dec_concat_after,
            positionwise_layer_type=dec_positionwise_layer_type,
            positionwise_conv_kernel_size=dec_positionwise_conv_kernel_size,
            macaron_style=dec_macaron_style,
            pos_enc_layer_type=dec_pos_enc_layer_type,
            selfattention_layer_type=dec_selfattention_layer_type,
            activation_type=dec_activation_type,
            use_cnn_module=dec_use_cnn_module,
            cnn_module_kernel=dec_cnn_module_kernel,
            padding_idx=dec_padding_idx,
        )

        self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(
            characters.squeeze(2), pos=pos_char, length=pos_spec
        )
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output = self.decoder(post_out, pos=pos_spec)
        output = self.postnet(mel_output)

        return output, None, mel_output, None


class ConformerSVS_FULL_combine(nn.Module):
    """Conformer Transformer Network."""

    def __init__(
        self,
        phone_size,
        singer_size,
        embed_size,
        output_dim,
        n_mels=-1,
        enc_attention_dim=256,
        enc_attention_heads=4,
        enc_linear_units=2048,
        enc_num_blocks=6,
        enc_dropout_rate=0.1,
        enc_positional_dropout_rate=0.1,
        enc_attention_dropout_rate=0.0,
        enc_input_layer="conv2d",
        enc_normalize_before=True,
        enc_concat_after=False,
        enc_positionwise_layer_type="linear",
        enc_positionwise_conv_kernel_size=1,
        enc_macaron_style=False,
        enc_pos_enc_layer_type="abs_pos",
        enc_selfattention_layer_type="selfattn",
        enc_activation_type="swish",
        enc_use_cnn_module=False,
        enc_cnn_module_kernel=31,
        enc_padding_idx=-1,
        dec_attention_dim=256,
        dec_attention_heads=4,
        dec_linear_units=2048,
        dec_num_blocks=6,
        dec_dropout_rate=0.1,
        dec_positional_dropout_rate=0.1,
        dec_attention_dropout_rate=0.0,
        dec_input_layer="conv2d",
        dec_normalize_before=True,
        dec_concat_after=False,
        dec_positionwise_layer_type="linear",
        dec_positionwise_conv_kernel_size=1,
        dec_macaron_style=False,
        dec_pos_enc_layer_type="abs_pos",
        dec_selfattention_layer_type="selfattn",
        dec_activation_type="swish",
        dec_use_cnn_module=False,
        dec_cnn_module_kernel=31,
        dec_padding_idx=-1,
        device="cuda",
    ):
        """init."""
        super(ConformerSVS_FULL_combine, self).__init__()
        self.encoder = Conformer_Encoder(
            phone_size,
            embed_size,
            attention_dim=enc_attention_dim,
            attention_heads=enc_attention_heads,
            linear_units=enc_linear_units,
            num_blocks=enc_num_blocks,
            dropout_rate=enc_dropout_rate,
            positional_dropout_rate=enc_positional_dropout_rate,
            attention_dropout_rate=enc_attention_dropout_rate,
            input_layer=enc_input_layer,
            normalize_before=enc_normalize_before,
            concat_after=enc_concat_after,
            positionwise_layer_type=enc_positionwise_layer_type,
            positionwise_conv_kernel_size=enc_positionwise_conv_kernel_size,
            macaron_style=enc_macaron_style,
            pos_enc_layer_type=enc_pos_enc_layer_type,
            selfattention_layer_type=enc_selfattention_layer_type,
            activation_type=enc_activation_type,
            use_cnn_module=enc_use_cnn_module,
            cnn_module_kernel=enc_cnn_module_kernel,
            padding_idx=enc_padding_idx,
        )
        self.enc_postnet = Encoder_Postnet_combine(embed_size, singer_size)

        self.decoder = Conformer_Decoder(
            embed_size,
            n_mels=n_mels,
            attention_dim=dec_attention_dim,
            attention_heads=dec_attention_heads,
            linear_units=dec_linear_units,
            num_blocks=dec_num_blocks,
            dropout_rate=dec_dropout_rate,
            positional_dropout_rate=dec_positional_dropout_rate,
            attention_dropout_rate=dec_attention_dropout_rate,
            input_layer=dec_input_layer,
            normalize_before=dec_normalize_before,
            concat_after=dec_concat_after,
            positionwise_layer_type=dec_positionwise_layer_type,
            positionwise_conv_kernel_size=dec_positionwise_conv_kernel_size,
            macaron_style=dec_macaron_style,
            pos_enc_layer_type=dec_pos_enc_layer_type,
            selfattention_layer_type=dec_selfattention_layer_type,
            activation_type=dec_activation_type,
            use_cnn_module=dec_use_cnn_module,
            cnn_module_kernel=dec_cnn_module_kernel,
            padding_idx=dec_padding_idx,
        )

        self.postnet = module.PostNet(n_mels, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        singer_vec,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(
            characters.squeeze(2), pos=pos_char, length=pos_spec
        )
        post_out = self.enc_postnet(
            encoder_out, phone, text_phone, pitch, beat, singer_vec
        )
        mel_output = self.decoder(post_out, pos=pos_spec)
        output = self.postnet(mel_output)

        return output, None, mel_output, None


# Reproduce the DAR model from USTC


class USTC_Prenet(nn.Module):
    """Singing Voice Synthesis Using Deep Autoregressive Neural Networks.

       for Acoustic Modeling from USTC, adapted by GS
    - herf: https://arxiv.org/pdf/1906.08977.pdf
    """

    def __init__(
        self,
        dim_input,
        multi_history_num=2,
        middle_dim=64,
        fc_drop_rate=0.75,
        prenet_drop_rate=0.1,
        kernel_size=2,
        n_blocks=3,
        n_heads=2,
        device="cuda",
    ):
        """init."""
        super(USTC_Prenet, self).__init__()

        self.multi_history_num = multi_history_num

        # FC & Dropout (ReLu in forward function)
        self.fc1 = nn.Linear(dim_input, middle_dim)
        self.dropout1 = nn.Dropout(fc_drop_rate)

        self.fc2 = nn.Linear(middle_dim, middle_dim)
        self.dropout2 = nn.Dropout(fc_drop_rate)

        # Conv1d & Barch Norm
        n_in = middle_dim
        n_out = middle_dim
        self.conv1d_and_BN = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size, stride=1, bias=True),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
        )

        # Postional Encoding Layer
        self.pos_code = module.PositionalEncoding(
            n_out
        )  # the same output dim as Layer Conv1d & Barch Norm
        self.fc_pos = nn.Linear(n_out, n_out)

        # Multi-head Self-Attention Layer
        self.self_attention_layers = nn.ModuleList()
        for i in range(n_blocks):
            self.self_attention_layers.append(
                module.MultiHeadAttentionLayer(
                    hid_dim=middle_dim,
                    n_heads=n_heads,
                    dropout=prenet_drop_rate,
                    device=device,
                )
            )

        # Final FC & Residual Connection
        self.fc3 = nn.Linear(middle_dim, middle_dim)

        self.multi_history_num = multi_history_num
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.device = device

    def forward(self, x):
        """forward."""
        # x: [batch size, multi_history_num, n_dim] - n_mel=40
        # in USTC model & with 1 energy

        # padding - same
        batch_size = np.shape(x)[0]

        pad_length = self.kernel_size - 1

        if pad_length == 1:
            x = torch.cat(
                (x, torch.zeros(batch_size, 1, self.dim_input).to(self.device)), dim=1
            )
        else:
            pad_before_length = pad_length // 2
            pad_after_length = pad_length - pad_before_length

            vector_pad_before = torch.zeros(
                batch_size, pad_before_length, self.dim_input
            ).to(self.device)
            vector_pad_after = torch.zeros(
                batch_size, pad_after_length, self.dim_input
            ).to(self.device)

            x = torch.cat((vector_pad_before, x, vector_pad_after), dim=1)

        # FC & Dropout (ReLu in forward function)
        output = self.dropout1(torch.relu(self.fc1(x)))
        output_fc = self.dropout2(
            torch.relu(self.fc2(output))
        )  # [batch, length, embed_size]

        # Conv1d & Barch Norm
        output_fc = output_fc.permute(0, 2, 1)  # [batch, embed_size, length]
        output = self.conv1d_and_BN(
            output_fc
        )  # output: [batch size, new_length, middle_dim] !!!
        output = output.permute(0, 2, 1)
        assert (
            np.shape(output)[1] == self.multi_history_num
        )  # assert conved length the same as before

        # Postional Encoding Layer
        pos = self.pos_code(torch.transpose(output, 0, 1))
        pos_encode = self.fc_pos(torch.transpose(pos, 0, 1))

        output = output + pos_encode  # output: [batch size, new_length, middle_dim]

        for layer in self.self_attention_layers:
            output, att = layer(query=output, key=output, value=output)

        output_fc = self.fc3(output)

        # Residual Connection
        output = output + output_fc  # output: [batch size, new_length, middle_dim]

        output = output.view(
            batch_size, -1
        )  # output: [batch size, new_length * middle_dim]

        return output


class USTC_SVS(nn.Module):
    """Singing Voice Synthesis Using Deep Autoregressive Neural Networks.

    for Acoustic Modeling from USTC, adapted by GS
    - herf: https://arxiv.org/pdf/1906.08977.pdf
    """

    def __init__(
        self,
        phone_size=87,
        embed_size=512,
        middle_dim_fc=512,
        output_dim=80,
        multi_history_num=2,
        middle_dim_prenet=64,
        n_blocks_prenet=3,
        n_heads_prenet=2,
        kernel_size_prenet=2,
        bi_d_model=256,
        bi_num_layers=1,
        uni_d_model=128,
        uni_num_layers=1,
        feedbackLink_drop_rate=0.75,
        dropout=0.1,
        device="cuda",
    ):
        """init."""
        super(USTC_SVS, self).__init__()

        self.emb_phone = nn.Embedding(phone_size, embed_size)

        self.fc_pitch = nn.Linear(
            1, embed_size
        )  # because pitch is not int | stair-like

        self.emb_beats = nn.Embedding(2, embed_size)  # only 0 and 1 two possibilities

        # FC (Tanh in forward function)
        self.fc1 = nn.Linear(embed_size * 3, middle_dim_fc)
        self.fc2 = nn.Linear(middle_dim_fc, middle_dim_fc)

        # Prenet Module
        self.prenet_module = USTC_Prenet(
            dim_input=output_dim,
            multi_history_num=multi_history_num,
            middle_dim=middle_dim_prenet,
            fc_drop_rate=feedbackLink_drop_rate,
            prenet_drop_rate=dropout,
            n_blocks=n_blocks_prenet,
            n_heads=n_heads_prenet,
            kernel_size=kernel_size_prenet,
            device=device,
        )

        # DAR model
        self.bi_GRU = nn.GRU(
            input_size=embed_size,
            hidden_size=bi_d_model,
            num_layers=bi_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        uni_input_size = 2 * bi_d_model + multi_history_num * middle_dim_prenet
        self.uni_GRU = nn.GRU(
            input_size=uni_input_size,
            hidden_size=uni_d_model,
            num_layers=uni_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

        self.fc_linear = nn.Linear(uni_d_model, output_dim)
        self.output_dim = output_dim
        self.uni_d_model = uni_d_model
        self.device = device
        self.multi_history_num = multi_history_num

    def forward(self, phone, pitch, beats, length, args):
        """forward."""
        # phone, pitch, beats = [batch size, len, 1]
        # target = [batch size, max_len, feat_dim]

        batch_size = np.shape(phone)[0]
        length_list = (
            torch.max(length, dim=1)[0].detach().cpu()
        )  # length = [batch size]
        max_length = args.num_frames

        phone_embedding = self.emb_phone(phone.squeeze(-1))
        pitch_embedding = torch.relu(self.fc_pitch(pitch))
        beats_embedding = self.emb_beats(beats.squeeze(-1))

        # print(np.shape(phone_embedding), np.shape(phone))
        # print(np.shape(pitch_embedding))
        # print(np.shape(beats_embedding))

        input_embedding = torch.cat(
            (phone_embedding, pitch_embedding, beats_embedding), dim=2
        )
        # input_embedding: [ batch size, len, embed_size * 3 ]

        # FC (Tanh in forward function)
        output_fc1 = torch.tanh(self.fc1(input_embedding))
        output_fc2 = torch.tanh(
            self.fc2(output_fc1)
        )  # output_fc2: [batch size, max_len, middle_dim_fc]

        packed_embed = nn.utils.rnn.pack_padded_sequence(
            output_fc2, length_list, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden_bi_GRU = self.bi_GRU(packed_embed)
        output_bi_GRU, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # [batch, seq_len, 2 * bi_d_model]
        hidden_bi_GRU = hidden_bi_GRU.view(
            batch_size, -1
        )  # [batch size, bi_num_layers * 2 * bi_d_model]

        # output_uni_GRU = torch.zeros
        # (batch_size, max_length, self.uni_d_model).to(self.device)
        output = torch.zeros(batch_size, max_length, self.output_dim).to(self.device)
        for i in range(torch.max(length_list)):
            step_output_bi_GRU = output_bi_GRU[:, i, :].unsqueeze(
                1
            )  # [batch size, 1, 2*bi_d_model]

            if i < self.multi_history_num:
                index_begin = 0
                # index_end = index_begin + self.multi_history_num - 1

                for shifted in range(self.multi_history_num):
                    if shifted == 0:
                        history_output = output[:, index_begin + shifted, :].unsqueeze(
                            1
                        )
                    else:
                        history_output = torch.cat(
                            (
                                history_output,
                                output[:, index_begin + shifted, :].unsqueeze(1),
                            ),
                            dim=1,
                        )
                assert np.shape(history_output)[1] == self.multi_history_num
            else:
                index_begin = i
                for shifted in range(self.multi_history_num):
                    if shifted == 0:
                        history_output = output[:, index_begin + shifted, :].unsqueeze(
                            1
                        )
                    else:
                        history_output = torch.cat(
                            (
                                history_output,
                                output[:, index_begin + shifted, :].unsqueeze(1),
                            ),
                            dim=1,
                        )
                assert np.shape(history_output)[1] == self.multi_history_num

            # history_output: [batch size, 2, feat_dim]

            step_output_prenet = self.prenet_module(
                history_output
            )  # [batch size, multi_history_num * middle_dim]
            step_output_prenet = step_output_prenet.unsqueeze(
                1
            )  # [batch size, 1, multi_history_num * middle_dim]

            step_input_uni_GRU = torch.cat(
                (step_output_bi_GRU, step_output_prenet), dim=2
            )
            # step_input_uni_GRU:
            # [batch size, 1, 2*bi_d_model + multi_history_num * middle_dim]

            if i == 0:
                step_output_uni_GRU, hidden_uni_GRU = self.uni_GRU(step_input_uni_GRU)
            else:
                step_output_uni_GRU, hidden_uni_GRU = self.uni_GRU(
                    step_input_uni_GRU, hidden_uni_GRU
                )
            # step_output_uni_GRU: [batch size, 1, uni_d_model]

            # Store Result of time i
            output[:, i, :] = self.fc_linear(
                step_output_uni_GRU.squeeze(1)
            )  # [batch size, output_dim]

        return output


def label_2_float(x, bits):
    """Label_2_float."""
    return 2 * x / (2 ** bits - 1.0) - 1.0


def decode_mu_law(y, mu, from_labels=True):
    """Decode_mu_law."""
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


class MelResNet(nn.Module):
    """MelResNet."""

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        """Init."""
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        """Forward."""
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class ResBlock(nn.Module):
    """ResBlock."""

    def __init__(self, dims):
        """Init."""
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        """Forward."""
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class Stretch2d(nn.Module):
    """Stretch2d."""

    def __init__(self, x_scale, y_scale):
        """Init."""
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        """Forward."""
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    """UpsampleNetwork."""

    def __init__(
        self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad
    ):
        """Init."""
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        """Forward."""
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent : -self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    """WaveRNN."""

    def __init__(
        self,
        rnn_dims,
        fc_dims,
        bits,
        pad,
        upsample_factors,
        feat_dims,
        compute_dims,
        res_out_dims,
        res_blocks,
        hop_length,
        sample_rate,
        mode="RAW",
    ):
        """Init."""
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == "RAW":
            self.n_classes = 2 ** bits
        elif self.mode == "MOL":
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        # List of rnns to call `flatten_parameters()` on
        self._to_flatten = []
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(
            feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad
        )
        self.line = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)

        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.num_params()

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x, mels):
        """Forward."""
        device = next(self.parameters()).device  # use same device as parameters
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        h2 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.line(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(
        self, mels, save_path: Union[str, Path], batched, target, overlap, mu_law
    ):
        """Generate."""
        self.eval()

        device = next(self.parameters()).device  # use same device as parameters

        mu_law = mu_law if self.mode == "RAW" else False

        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            mels = torch.as_tensor(mels, device=device)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side="both")
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims, device=device)
            h2 = torch.zeros(b_size, self.rnn_dims, device=device)
            x = torch.zeros(b_size, 1, device=device)

            d = self.aux_dims
            aux_split = [aux[:, :, d * i : d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.line(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == "MOL":
                    sample = sample_from_discretized_mix_logistic(
                        logits.unsqueeze(0).transpose(1, 2)
                    )
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    x = sample.transpose(0, 1)

                elif self.mode == "RAW":
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length :] *= fade_out

        self.train()

        return output

    def get_gru_cell(self, gru):
        """Get_gru_cell."""
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side="both"):
        """Pad_tensor."""
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == "both" else t + pad
        padded = torch.zeros(b, total, c, device=x.device)
        if side == "before" or side == "both":
            padded[:, pad : pad + t, :] = x
        elif side == "after":
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        """Fold the tensor with overlap for quick batched inference.

            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """
        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side="after")

        folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):
        """Apply a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        """
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)
        linear = np.ones((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([linear, fade_out])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self):
        """Get_step."""
        return self.step.data.item()

    def log(self, path, msg):
        """Log."""
        with open(path, "a") as f:
            print(msg, file=f)

    def load(self, path: Union[str, Path]):
        """Load."""
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        """Save."""
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        """Num_params."""
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters

    def _flatten_parameters(self):
        """Call `flatten_parameters` on all the rnns used by the WaveRNN.

        Used to improve efficiency and avoid PyTorch yelling at us.
        """
        [m.flatten_parameters() for m in self._to_flatten]


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """Sample from discretized mixture of logistic distributions.

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix : 2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(
        torch.sum(y[:, :, 2 * nr_mix : 3 * nr_mix] * one_hot, dim=-1), min=log_scale_min
    )
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))

    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)

    return x


def _test():
    """test."""
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

    # hidden_size = 256
    # embed_size = 256
    # nhead = 4
    # dropout = 0.1
    # activation = 'relu'
    # glu_kernel = 3
    # num_dec_block = 3
    # glu_num_layers = 1
    # num_glu_block = 3

    # #test encoder and encoer_postnet
    # encoder = Encoder(phone_size, embed_size, hidden_size, dropout,
    # num_glu_block, num_layers=glu_num_layers, glu_kernel=glu_kernel)
    # encoder_out, text_phone = encoder(phone.squeeze(2))
    # print('encoder_out.size():',encoder_out.size())

    # post = Encoder_Postnet(embed_size)
    # post_out = post(encoder_out, phone, text_phone, pitch.float(), beat)
    # print('post_net_out.size():',post_out.size())

    # # test model as a whole
    # # model = GLU_Transformer(phone_size, hidden_size, embed_size,
    # glu_num_layers, dropout, num_dec_block, nhead, feat_dim)
    # spec_pred = model(char, phone, pitch, beat, src_key_padding_mask=seq_len,
    # char_key_padding_mask=char_seq_len)
    # # print(spec_pred)

    # # test decoder
    # out_from_encoder = torch.zeros(batch_size, max_length, hidden_size)
    # for i in range(batch_size):
    #     length = seq_len_list[i]
    #     out_from_encoder[i, :length, :] = torch.randn(length, hidden_size)
    # decoder = Decoder(num_dec_block, embed_size, feat_dim, nhead, dropout)
    # decoder_out, att =
    # decoder(out_from_encoder, src_key_padding_mask=seq_len)
    # print(decoder_out.size())
    # print(att.size())


if __name__ == "__main__":
    _test()
