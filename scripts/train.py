# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score

import torch.optim as optim

from net import *
from util import *


def main():
    parser = argparse.ArgumentParser(description='pytorch efficient architecture')

    parser.add_argument('--batch', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'cifar10', 'stl10', 'food101'],
                        help='The name of using dataset.')
    parser.add_argument('--block', default='plain',
                        choices=['plain', 'residual', 'residual_bottleneck', 'resnext', 'xception', 'clc', 'dense'],
                        help='The type of using block.')

    args = parser.parse_args()
    print(args)

    batch_size = args.batch
    epoch_size = args.epoch
    dataset_name = args.dataset
    block_type = args.block

    if block_type == 'plain':
        block = DoubleBlock
    elif block_type == 'residual':
        block = ResidualBlock
    elif block_type == 'residual_bottleneck':
        block = ResidualBottleneckBlock
    elif block_type == 'resnext':
        block = ResNeXtBlock
    elif block_type == 'xception':
        block = XceptionBlock
    elif block_type == 'clc':
        block = CLCBlock
    elif block_type == 'dense':
        block = DenseBlock

    train_set, valid_set, ch_list = create_dataset(dataset_name)
    print('training data size: ', len(train_set))
    print('validation data size: ', len(valid_set))
    display_interval = (len(train_set) / batch_size) // 10

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    model = BaseArchitecture(block, ch_list)
    model.apply(weights_init)
    print(model)
    print()

    model.to(device)  # for GPU

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, epoch_size // 2)

    epoch_list = []
    train_acc_list, train_loss_list, train_time_mean_list, train_time_std_list = [], [], [], []
    valid_acc_list, valid_loss_list, valid_time_mean_list, valid_time_std_list = [], [], [], []
    for epoch in range(epoch_size):  # loop over the dataset multiple times
        scheduler.step()

        model.train()
        train_true, train_pred = [], []
        running_loss, train_loss = 0., 0.
        inference_time = []
        for itr, data in enumerate(train_loader, 0):
            images, labels = data

            train_true.extend(labels.tolist())

            images, labels = images.to(device), labels.to(device)  # for GPU

            optimizer.zero_grad()

            start_time = time.perf_counter()
            logits = model(images)
            end_time = time.perf_counter()
            inference_time.append(end_time - start_time)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            train_pred.extend(predicted.tolist())

            running_loss += loss.item()
            train_loss += loss.item()
            if itr % display_interval == display_interval - 1:  # print every 100 mini-batches
                print('    [epochs: {}, mini-batches: {}, images: {}] loss: {:.6f}'.format(
                    epoch + 1, itr + 1, (itr + 1) * batch_size, running_loss / display_interval))
                running_loss = 0.0

        train_acc_list.append(accuracy_score(train_true, train_pred))
        train_loss_list.append(train_loss / (itr + 1))
        train_time_mean_list.append(np.mean(inference_time))
        train_time_std_list.append(np.std(inference_time))

        model.eval()
        valid_true, valid_pred = [], []
        valid_loss = 0.
        inference_time = []
        for itr, data in enumerate(valid_loader):
            images, labels = data
            valid_true.extend(labels.tolist())
            images, labels = images.to(device), labels.to(device)  # for GPU

            start_time = time.perf_counter()
            with torch.no_grad():
                logits = model(images)
            end_time = time.perf_counter()
            inference_time.append(end_time - start_time)

            loss = criterion(logits, labels)

            _, predicted = torch.max(logits.data, 1)
            valid_pred.extend(predicted.tolist())

            valid_loss += loss.item()

        valid_acc_list.append(accuracy_score(valid_true, valid_pred))
        valid_loss_list.append(valid_loss / (itr + 1))
        valid_time_mean_list.append(np.mean(inference_time))
        valid_time_std_list.append(np.std(inference_time))

        print('epocs: {}'.format(epoch + 1))

        print('train  acc.: {:.3f}  loss: {:.6f}  time: {:.2e} +/- {:.2e}'.format(
            train_acc_list[-1], train_loss_list[-1], train_time_mean_list[-1], train_time_std_list[-1]))

        print('valid  acc.: {:.3f}  loss: {:.6f}  time: {:.2e} +/- {:.2e}'.format(
            valid_acc_list[-1], valid_loss_list[-1], valid_time_mean_list[-1], valid_time_std_list[-1]))

        print()

        epoch_list.append(epoch + 1)

    print('Finished Training')

    print('Save Network')
    torch.save(model.state_dict(), '../models/{}_{}.pth'.format(dataset_name, block_type))

    df = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list,
                       'valid/accuracy': valid_acc_list,
                       'train/loss': train_loss_list,
                       'valid/loss': valid_loss_list,
                       'train/time_mean': train_time_mean_list,
                       'train/time_std': train_time_std_list,
                       'valid/time_mean': valid_time_mean_list,
                       'valid/time_std': valid_time_std_list})

    print('Save Training Log')
    df.to_csv('../logs/{}_{}.csv'.format(dataset_name, block_type), index=False)


if __name__ == '__main__':
    main()
