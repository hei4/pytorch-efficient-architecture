# -*- coding: utf-8 -*-

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
    :param m: ニューラルネットワークを構成する層
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            # 畳み込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:     # バッチノーマライゼーションの場合
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 101)
        self.initialization()

    def forward(self, x):
        h = self.pool(F.relu(self.cv1(x)))
        h = self.pool(F.relu(self.cv2(h)))
        h = h.view(-1, 64 * 56 * 56)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def initialization(self):
        nn.init.xavier_normal_(self.cv1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.cv2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.fc2.weight, gain=np.sqrt(2))

################################################################
# NEURAL NETWORKS
################################################################
class PlainArchitecture(nn.Module):
    def __init__(self):
        super(PlainArchitecture, self).__init__()
        self.blocks = nn.Sequential(
            SingleBlock(3, 64, stride=2),

            DoubleBlock(64, 128, first_stride=2),
            DoubleBlock(128, 128),

            DoubleBlock(128, 256, first_stride=2),
            DoubleBlock(256, 256),

            DoubleBlock(256, 512, first_stride=2),
            DoubleBlock(512, 512),

            DoubleBlock(512, 1024, first_stride=2),
            DoubleBlock(1024, 1024),

            GAPBlock(1024, 101)
        )

    def forward(self, x):
        y = self.blocks(x)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.blocks = nn.Sequential(
            SingleBlock(3, 64, stride=2),

            ResidualBlock(64, 128, first_stride=2),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, first_stride=2),
            ResidualBlock(256, 256),

            ResidualBlock(256, 512, first_stride=2),
            ResidualBlock(512, 512),

            ResidualBlock(512, 1024, first_stride=2),
            ResidualBlock(1024, 1024),

            GAPBlock(1024, 101)
        )

    def forward(self, x):
        y = self.blocks(x)
        return y


################################################################
# BLOCKS
################################################################
class GAPBlock(nn.Module):
    def __init__(self, hidden_ch, out_ch):
        super(GAPBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_ch, out_ch)
        self.hidden_ch = hidden_ch

    def forward(self, x):
        h = self.gap(x)
        h = h.view([-1, self.hidden_ch])
        y = self.fc(h)
        return y


class SingleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(SingleBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DoubleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(DoubleBlock, self).__init__()
        self.layers = nn.ModuleList([
            SingleBlock(in_ch, out_ch, first_stride),
            SingleBlock(out_ch, out_ch)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_0 = SingleBlock(in_ch, out_ch, first_stride)
        self.conv_1 = SingleBlock(out_ch, out_ch)

        if first_stride != 1:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0)
        self.first_stride = first_stride

    def forward(self, x):
        h = self.conv_0(x)
        h = self.conv_1(h)

        if self.first_stride != 1:
            x = self.shortcut(x)

        return h + x

