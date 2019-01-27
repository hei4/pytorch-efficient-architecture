# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


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
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DoubleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(DoubleBlock, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
        )

        self.layer2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        if first_stride != 1:
            self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0)
        self.first_stride = first_stride

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)

        if self.first_stride != 1:
            x = self.downsample(x)

        return h + x


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(ResidualBottleneckBlock, self).__init__()
        intermediate_ch = out_ch // 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, intermediate_ch, kernel_size=1, stride=first_stride, padding=0),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(intermediate_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        if first_stride != 1:
            self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0)
        self.first_stride = first_stride

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        if self.first_stride != 1:
            x = self.downsample(x)

        return h + x


class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(ResNeXtBlock, self).__init__()
        intermediate_ch = out_ch // 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, intermediate_ch, kernel_size=1, stride=first_stride, padding=0),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(intermediate_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        if first_stride != 1:
            self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0)
        self.first_stride = first_stride

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        if self.first_stride != 1:
            x = self.downsample(x)

        return h + x


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(XceptionBlock, self).__init__()
        self.layer1 = nn.Sequential(
            # point-wise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # depth-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # point-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # depth-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        if first_stride != 1:
            self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride, padding=0)
        self.first_stride = first_stride

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)

        if self.first_stride != 1:
            x = self.downsample(x)

        return h + x


class CLCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_stride=1):
        super(CLCBlock, self).__init__()

        if out_ch == 128:
            g1, g2 = 32, 4
        elif out_ch == 256:
            g1, g2 = 64, 4
        elif out_ch == 512:
            g1, g2 = 64, 8
        elif out_ch == 1024:
            g1, g2 = 128, 8

        first_g1 = g1 // first_stride

        self.inplaned_layers1 = nn.ModuleList([
            nn.Conv2d(in_ch, first_g1, kernel_size=3, stride=first_stride, padding=1, groups=first_g1) for _ in range(g2)
        ])

        self.gruop_layer1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=g2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.inplaned_layers2 = nn.ModuleList([
            nn.Conv2d(out_ch, g1, kernel_size=3, stride=1, padding=1, groups=g1) for _ in range(g2)
        ])

        self.gruop_layer2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=g2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        h_list = []
        for inplaned_layer in self.inplaned_layers1:
            h_list.append(inplaned_layer(x))

        h = torch.cat(h_list, dim=1)
        h = self.gruop_layer1(h)

        h_list = []
        for inplaned_layer in self.inplaned_layers2:
            h_list.append(inplaned_layer(h))

        h = torch.cat(h_list, dim=1)
        h = self.gruop_layer2(h)

        return h


################################################################
# NEURAL NETWORKS
################################################################
class BaseArchitecture(nn.Module):
    def __init__(self, block_class, ch_list):
        super(BaseArchitecture, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(ch_list)-1):
            print('blocks[{}]: {} -> {}'.format(i, ch_list[i], ch_list[i+1]))

            if i == 0:
                self.blocks.append(SingleBlock(ch_list[i], ch_list[i+1], stride=2))
            elif i == len(ch_list) - 2:
                self.blocks.append(GAPBlock(ch_list[i], ch_list[i+1]))
            elif i % 2 == 1:
                self.blocks.append(block_class(ch_list[i], ch_list[i + 1], first_stride=2))
            else:
                self.blocks.append(block_class(ch_list[i], ch_list[i + 1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
