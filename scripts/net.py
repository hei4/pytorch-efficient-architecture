# -*- coding: utf-8 -*-
import torch
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
        # m.bias.data.fill_(0)
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
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DoubleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(DoubleBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=downsample_rate, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=downsample_rate, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if downsample_rate != 1:
            self.transition_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)

        if self.downsample_rate != 1:
            x = self.transition_layer(x)

        return F.relu(h + x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(BottleneckBlock, self).__init__()
        intermediate_ch = out_ch // 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, intermediate_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(intermediate_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        if downsample_rate != 1:
            self.transition_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        if self.downsample_rate != 1:
            x = self.transition_layer(x)

        return F.relu(h + x)


class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(ResNeXtBlock, self).__init__()
        intermediate_ch = out_ch // 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, intermediate_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(intermediate_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if downsample_rate != 1:
            self.transition_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        if self.downsample_rate != 1:
            x = self.transition_layer(x)

        return F.relu(h + x)


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(XceptionBlock, self).__init__()
        self.layer1 = nn.Sequential(
            # point-wise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            # depth-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # point-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            # depth-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if downsample_rate != 1:
            self.transition_layer1 = nn.Sequential(
                nn.MaxPool2d(downsample_rate)
            )
        
            self.transition_layer2 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=downsample_rate, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)

        if self.downsample_rate != 1:
            h = self.transition_layer1(h)
            x = self.transition_layer2(x)

        return F.relu(h + x)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate, growth_rate):
        super(DenseBlock, self).__init__()
        intermediate_out = (out_ch - in_ch) // growth_rate
        self.growth_rate = growth_rate

        self.dense_layers = nn.ModuleList()
        for i in range(growth_rate):
            intermediate_in = in_ch + i * intermediate_out

            self.dense_layers.add_module('block_{}'.format(i), nn.Sequential(
                nn.Conv2d(intermediate_in, intermediate_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(intermediate_out),
                nn.ReLU(),
                nn.Conv2d(intermediate_out, intermediate_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(intermediate_out),
                nn.ReLU(),
            ))

        self.transition_layer = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(downsample_rate)
        )

    def forward(self, x):
        for dense_layer in self.dense_layers:
            h = dense_layer(x)
            x = torch.cat([x, h], dim=1)

        h = self.transition_layer(x)
        return h


class MobileV1Block(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(MobileV1Block, self).__init__()
        self.layer1 = nn.Sequential(
            # depth-wise
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=downsample_rate, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # point-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # depth-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # point-wise
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)

        return h


class MobileV2Block(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1, expansion_factor=6):
        super(MobileV2Block, self).__init__()
        intermediate1_ch = in_ch * expansion_factor
        intermediate2_ch = out_ch * expansion_factor

        self.layer1 = nn.Sequential(
            # expansion
            nn.Conv2d(in_ch, intermediate1_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate1_ch),
            nn.ReLU6(),
            # depth-wise
            nn.Conv2d(intermediate1_ch, intermediate1_ch, kernel_size=3, stride=downsample_rate, padding=1, bias=False,
                      groups=intermediate1_ch),
            nn.BatchNorm2d(intermediate1_ch),
            nn.ReLU6(),
            # projection
            nn.Conv2d(intermediate1_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.layer2 = nn.Sequential(
            # expansion
            nn.Conv2d(out_ch, intermediate2_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate2_ch),
            nn.ReLU6(),
            # depth-wise
            nn.Conv2d(intermediate2_ch, intermediate2_ch, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=intermediate2_ch),
            nn.BatchNorm2d(intermediate2_ch),
            nn.ReLU6(),
            # projection
            nn.Conv2d(intermediate2_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        if self.downsample_rate == 1:
            h = h + x
        x = F.relu6(h)

        h = self.layer2(x)
        h = h + x

        return F.relu6(h)


class ShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1, groups=4):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
        intermediate_ch = out_ch // 4
        if downsample_rate == 1:
            intermediate_out = out_ch
        else:
            intermediate_out = out_ch - in_ch

        self.layer1 = nn.Sequential(
            # group
            nn.Conv2d(in_ch, intermediate_ch, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # depth-wise
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=downsample_rate, padding=1, bias=False,
                      groups=intermediate_ch),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(),
            # group
            nn.Conv2d(intermediate_ch, intermediate_out, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(intermediate_out),
        )

        self.layer3 = nn.Sequential(
            # group
            nn.Conv2d(out_ch, intermediate_ch, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            # depth-wise
            nn.Conv2d(intermediate_ch, intermediate_ch, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=intermediate_ch),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(),
            # group
            nn.Conv2d(intermediate_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(out_ch),
        )

        if downsample_rate != 1:
            self.transition_layer = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_rate = downsample_rate

    def forward(self, x):
        h = self.layer1(x)
        h = self.channel_shuffle(h)

        h = self.layer2(h)

        if self.downsample_rate == 1:
            h = h + x
        else:
            x = self.transition_layer(x)
            h = torch.cat([h, x], dim=1)
        x = F.relu(h)

        h = self.layer3(x)
        h = self.channel_shuffle(h)

        h = self.layer4(h)
        h = h + x

        return F.relu(h)

    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1, self.groups, H, W)
        x = x.transpose(1, 2)
        return torch.flatten(x, 1, 2)


'''
class CLCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_rate=1):
        super(CLCBlock, self).__init__()

        if out_ch == 128:
            g1, g2 = 32, 4
        elif out_ch == 256:
            g1, g2 = 64, 4
        elif out_ch == 512:
            g1, g2 = 64, 8
        elif out_ch == 1024:
            g1, g2 = 128, 8

        first_g1 = g1 // downsample_rate
        self.inplaned_layer1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=downsample_rate, padding=1, groups=first_g1, bias=False)
        )

        self.index_list1 = []
        for i in range(g2):
            self.index_list1.append(list(range(i, in_ch, g2)))

        self.gruop_layer1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=g2, bias=False)
        )

        self.inplaned_layer2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=g1, bias=False)
        )

        self.index_list2 = []
        for i in range(g2):
            self.index_list2.append(list(range(i, out_ch, g2)))

        self.gruop_layer2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=g2, bias=False),
        )

    def forward(self, x):
        h = self.inplaned_layer1(x)

        h_list = []
        for index in self.index_list1:
            h_list.append(h[:, index, :, :])
        h = torch.cat(h_list, dim=1)

        h = self.gruop_layer1(h)

        h = self.inplaned_layer2(h)

        h_list = []
        for index in self.index_list2:
            h_list.append(h[:, index, :, :])
        h = torch.cat(h_list, dim=1)

        h = self.gruop_layer2(h)

        return h
'''


################################################################
# NEURAL NETWORKS
################################################################
def create_stage(block_class, in_ch, out_ch, n_blocks=4):
    stage = nn.Sequential()

    if block_class == DenseBlock:
        stage.add_module('stage_dense', block_class(in_ch, out_ch, downsample_rate=2, growth_rate=2*n_blocks))

    else:
        for i in range(n_blocks):
            if i == 0:
                stage.add_module('block_{}'.format(i), block_class(in_ch, out_ch, downsample_rate=2))
            else:
                stage.add_module('block_{}'.format(i), block_class(out_ch, out_ch))

    return stage


class BaseArchitecture(nn.Module):
    def __init__(self, block_class, ch_list):
        super(BaseArchitecture, self).__init__()
        self.stages = nn.ModuleList()
        for i in range(len(ch_list)-1):
            print('blocks[{}]: {} -> {}'.format(i, ch_list[i], ch_list[i+1]))

            if i == 0:
                self.stages.append(SingleBlock(ch_list[i], ch_list[i+1], stride=2))
            elif i == len(ch_list) - 2:
                self.stages.append(GAPBlock(ch_list[i], ch_list[i+1]))
            else:
                stage = create_stage(block_class, ch_list[i], ch_list[i+1])
                self.stages.append(stage)

    def forward(self, x):
        for block in self.stages:
            x = block(x)
        return x
