# -*- coding: utf-8 -*-
import os
import pandas as pd
from matplotlib import pyplot as plt

cmap = plt.get_cmap('tab10')

dataset_list = ['mnist', 'cifar10', 'stl10']
block_list = ['plain', 'residual', 'bottleneck', 'resnext',
              'xception', 'dense', 'mobile_v1', 'shuffle']

for dataset_name in dataset_list:
    df_dict = {}
    size_dict = {}
    for block_type in block_list:
        df_dict[block_type] = pd.read_csv('../logs/{}_{}.csv'.format(dataset_name, block_type))
        size_dict[block_type] = os.path.getsize('../models/{}_{}.pth'.format(dataset_name, block_type))

    # model size
    fig, ax = plt.subplots()
    for block_type in block_list:
        ax.bar(block_type, size_dict[block_type]/1024, label=block_type)
    ax.set_xlabel('Block type')
    ax.set_ylabel('File size [KB]')
    ax.tick_params(labelbottom=False)
    ax.legend()
    plt.title('file sizes of {} trained models'.format(dataset_name))
    plt.savefig('../images/{}_size.png'.format(dataset_name))
    plt.close()

    # train/acc.
    fig, ax = plt.subplots()
    for block_type in block_list:
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['train/accuracy'], label=block_type)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.title('{} training/accuracy'.format(dataset_name))
    plt.savefig('../images/{}_train_acc.png'.format(dataset_name))
    plt.close()

    _, _, ymin, ymax = ax.axis()

    # valid/acc.
    fig, ax = plt.subplots()
    for block_type in block_list:
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['valid/accuracy'], label=block_type)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.title('{} validation/accuracy'.format(dataset_name))
    plt.savefig('../images/{}_valid_acc.png'.format(dataset_name))
    plt.close()

    # train/loss
    fig, ax = plt.subplots()
    for block_type in block_list:
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['train/loss'], label=block_type)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title('{} training/loss'.format(dataset_name))
    plt.savefig('../images/{}_train_loss.png'.format(dataset_name))
    plt.close()

    _, _, ymin, ymax = ax.axis()

    # valid/loss
    fig, ax = plt.subplots()
    for block_type in block_list:
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['valid/loss'], label=block_type)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title('{} validation/loss'.format(dataset_name))
    plt.savefig('../images/{}_valid_loss.png'.format(dataset_name))
    plt.close()

    # train/time
    fig, ax = plt.subplots()
    for i, block_type in enumerate(block_list):
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['train/time_mean'], label=block_type, color=cmap(i))
        ax.fill_between(df_dict[block_type]['epoch'],
                        df_dict[block_type]['train/time_mean'] + df_dict[block_type]['train/time_std'],
                        df_dict[block_type]['train/time_mean'] - df_dict[block_type]['train/time_std'],
                        color=cmap(i), alpha=0.2)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Forward propergation time [s]')
    ax.legend()
    plt.title('{} training/forward propergation time'.format(dataset_name))
    plt.savefig('../images/{}_train_time.png'.format(dataset_name))
    plt.close()

    _, _, ymin, ymax = ax.axis()

    # valid/time
    fig, ax = plt.subplots()
    for i, block_type in enumerate(block_list):
        ax.plot(df_dict[block_type]['epoch'], df_dict[block_type]['valid/time_mean'], label=block_type, color=cmap(i))
        ax.fill_between(df_dict[block_type]['epoch'],
                        df_dict[block_type]['valid/time_mean'] + df_dict[block_type]['valid/time_std'],
                        df_dict[block_type]['valid/time_mean'] - df_dict[block_type]['valid/time_std'],
                        color=cmap(i), alpha=0.2)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Forward propergation time [s]')
    ax.legend()
    plt.title('{} validation/forward propergation time'.format(dataset_name))
    plt.savefig('../images/{}_valid_time.png'.format(dataset_name))
    plt.close()
