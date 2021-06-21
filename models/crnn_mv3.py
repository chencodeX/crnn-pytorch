#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen
'''
@Author : zihao.chen
@File : crnn_mv3.py 
@Create Date : 2021/6/2
@Descirption :
'''

from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import init
from models.netmodule import CALayer, BidirectionalLSTM


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(self.avg_pool(x))


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, ):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, (2, 1)),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), (2, 1)),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), (2, 1)),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self):
        super(MobileNetV3_Small, self).__init__()

        self.conv1_3x3_3 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_5x5_3 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 32, 32, 32, nn.ReLU(inplace=True), SeModule(32), (2,1)),
            Block(3, 32, 72, 40, nn.ReLU(inplace=True), None, 1),
            Block(3, 40, 120, 40, nn.ReLU(inplace=True), None, (2,1)),
            Block(5, 40, 240, 80, hswish(), SeModule(80), 1),
            Block(5, 80, 200, 80, hswish(), SeModule(80), 1),
            Block(5, 80, 240, 80, hswish(), SeModule(80), 2),
            Block(5, 80, 184, 80, hswish(), SeModule(80), 1),
            Block(5, 80, 480, 112, hswish(), SeModule(112), (2,1)),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 320, hswish(), SeModule(320), 1),
            Block(5, 320, 960, 480, hswish(), SeModule(480), 1),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1_3x3_3(x)
        x2 = self.conv1_5x5_3(x)
        x = torch.cat((x1, x2), dim=1)
        out = self.hs1(self.bn1(x))
        out = self.bneck(out)
        return out


def mobilenet_v3_small(pretrained, is_gray=False, **kwargs):
    model = MobileNetV3_Small()
    if pretrained:
        pretrained_dict = torch.load('./pre_model/mbv3_small.old.pth.tar')['state_dict']

        state = model.state_dict()
        for key in state.keys():
            if 'module.' + key in pretrained_dict.keys():
                if (key == 'conv1.weight' and is_gray):
                    state[key] = torch.mean(pretrained_dict['module.' + key], 1).unsqueeze(1)
                else:
                    state[key] = pretrained_dict['module.' + key]
        model.load_state_dict(state)
    return model


class CRNN_Head(nn.Module):
    def __init__(self,
                 use_dynamic=False,
                 lstm_num=2,
                 inchannel=160,
                 hiddenchannel=256,
                 classes=4000 + 1):
        super(CRNN_Head, self).__init__()

        self.lstm_num = lstm_num

        self.attention = SeModule(inchannel)
        self.use_dynamic = use_dynamic

        assert lstm_num > 0, Exception('lstm_num need to more than 0 if use_lstm = True')
        for i in range(lstm_num):
            if (i == 0):
                if (lstm_num == 1):
                    setattr(self, 'lstm_{}'.format(i + 1),
                            BidirectionalLSTM(inchannel, hiddenchannel, classes, 0.5, dynamic=self.use_dynamic))
                else:
                    setattr(self, 'lstm_{}'.format(i + 1),
                            BidirectionalLSTM(inchannel, hiddenchannel, hiddenchannel, 0.5,
                                              dynamic=self.use_dynamic))
            elif (i == lstm_num - 1):
                setattr(self, 'lstm_{}'.format(i + 1),
                        BidirectionalLSTM(hiddenchannel, hiddenchannel, classes, 0.5, dynamic=self.use_dynamic))
            else:
                setattr(self, 'lstm_{}'.format(i + 1),
                        BidirectionalLSTM(hiddenchannel, hiddenchannel, hiddenchannel, 0.5,
                                          dynamic=self.use_dynamic))

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, seq_lens=None):
        b, c, h, w = x.size()
        # print(x.size())
        # assert h == 1, "the height of conv must be 1"

        x = self.attention(x)
        # print(x.size())
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w, b, c]

        if self.use_dynamic:
            assert seq_lens != None
        if self.use_dynamic:
            seq_lengths = torch.LongTensor(seq_lens.long())
            sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
            x = x[:, indices]
            _, desorted_indices = torch.sort(indices, descending=False)

        for i in range(self.lstm_num):
            if self.use_dynamic:
                x = getattr(self, 'lstm_{}'.format(i + 1))(x, sorted_seq_lengths)
            else:
                x = getattr(self, 'lstm_{}'.format(i + 1))(x)
        if self.use_dynamic:
            x = x[:, desorted_indices]
        return x


class CRNN(nn.Module):
    def __init__(self, hiddenchannel, classes, dynamic=False):
        super(CRNN, self).__init__()
        self.cnn_bo = MobileNetV3_Small()
        self.lstm_r = CRNN_Head(inchannel=480, hiddenchannel=hiddenchannel, classes=classes, use_dynamic=dynamic)
        self.dynamic = dynamic

    def forward(self, x, seq_lens=None):
        print('=='*10)
        x = self.cnn_bo(x)
        print(x.size())
        if self.dynamic:
            x = self.lstm_r(x, seq_lens)
        else:
            x = self.lstm_r(x)
        output = F.log_softmax(x, dim=2)
        return output
