# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score

import torch.optim as optim

from net import *
from util import *

import torch.nn.functional as F

F.conv2d

def main():
    parser = argparse.ArgumentParser(description='pytorch efficient architecture')

    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'cifar10', 'stl10', 'food101'],
                        help='The name of using dataset.')
    parser.add_argument('--block', default='plain',
                        choices=['plain', 'residual', 'bottleneck', 'resnext',
                                 'xception', 'dense', 'mobile_v1', 'mobile_v2', 'shuffle'],
                        help='The type of using block.')

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    block_type = args.block

    if block_type == 'plain':
        block = DoubleBlock
    elif block_type == 'residual':
        block = ResidualBlock
    elif block_type == 'bottleneck':
        block = BottleneckBlock
    elif block_type == 'resnext':
        block = ResNeXtBlock
    elif block_type == 'xception':
        block = XceptionBlock
    elif block_type == 'dense':
        block = DenseBlock
    elif block_type == 'mobile_v1':
        block = MobileV1Block
    elif block_type == 'mobile_v2':
        block = MobileV2Block
    elif block_type == 'shuffle':
        block = ShuffleBlock

    trainset, _, ch_list = create_dataset(dataset_name)

    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device('cpu')
    print('device:', gpu_device, cpu_device)

    model = BaseArchitecture(block, ch_list)
    model.apply(weights_init)
    print(model)
    print()

    model.to(gpu_device)  # for GPU

    param = torch.load('../models/{}_{}.pth'.format(dataset_name, block_type))
    model.load_state_dict(param)

    image = trainset[0][0]
    C, H, W = image.shape

    gpu_time = []
    for i in range(1000):
        x = torch.randn(1, C, H, W).to(gpu_device)

        start_time = time.perf_counter()
        _ = model(x)
        end_time = time.perf_counter()
        gpu_time.append(end_time - start_time)

    model.to(cpu_device)

    cpu_time = []
    for i in range(1000):
        x = torch.randn(1, C, H, W)

        start_time = time.perf_counter()
        _ = model(x)
        end_time = time.perf_counter()
        cpu_time.append(end_time - start_time)

    print('GPU: {} +/- {}'.format(np.mean(gpu_time), np.std(gpu_time)))
    print('CPU: {} +/- {}'.format(np.mean(cpu_time), np.std(cpu_time)))

    df = pd.DataFrame({'gpu/time': gpu_time,
                       'cpu/time': cpu_time})

    print('Save Training Log')
    df.to_csv('../logs/time/{}_{}.csv'.format(dataset_name, block_type), index=False)


if __name__ == '__main__':
    main()
