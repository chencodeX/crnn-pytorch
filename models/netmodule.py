#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen
'''
@Author : zihao.chen
@File : netmodule.py 
@Create Date : 2021/6/2
@Descirption :
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# class BidirectionalLSTM(torch.jit.ScriptModule):
class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut, drop_rate, dynamic=False):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.rnn.flatten_parameters()
        self.embedding = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(nHidden * 2, nOut))
        self.dynamic = dynamic

    # @torch.jit.script_method
    def forward(self, input, seq_lens):
        # type:  (Tensor,Tensor) -> Tensor
        # return:
        # self.rnn.flatten_parameters()
        if self.dynamic:
            # if enforce_sorted:
            # input and seq_lens need sort
            packed_seqs = pack_padded_sequence(input, seq_lens)
            # else:
            #     packed_seqs = pack_padded_sequence(input, seq_lens, enforce_sorted=False)
            out_dynamic, _ = self.rnn(packed_seqs)
            # print(out_dynamic.data.size())
            recurrent, lens = pad_packed_sequence(out_dynamic)

        else:
            recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
