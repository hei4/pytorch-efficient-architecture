# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

from net import *


def main():
    parser = argparse.ArgumentParser(description='pytorch efficient architecture')
    parser.add_argument('--batch', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--display', '-d', type=int, default=100,
                        help='Number of interval to show progress')
    args = parser.parse_args()

    batch_size = args.batch
    epoch_size = args.epoch
    display_interval = args.display

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    data_set = torchvision.datasets.ImageFolder(root='/mnt/hdd/sika/data/food-101/images', transform=train_transform)

    data_size = len(data_set)
    train_size = int(data_size * 0.8)
    valid_size = data_size - train_size

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

    print(data_size)
    print(train_size)
    print(valid_size)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_set.transform = train_transform
    print(train_set.transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    valid_set.transform = valid_transform
    print(valid_set.transform)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # model = PlainArchitecture()
    model = ResNet()

    model.apply(weights_init)
    print(model)
    print()

    model.to(device)  # for GPU

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    epoch_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(epoch_size):  # loop over the dataset multiple times
        running_loss = 0.0

        model.train()
        train_true = []
        train_pred = []
        for itr, data in enumerate(train_loader, 0):
            images, labels = data

            train_true.extend(labels.tolist())

            images, labels = images.to(device), labels.to(device)  # for GPU

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            train_pred.extend(predicted.tolist())

            running_loss += loss.item()
            if itr % display_interval == display_interval - 1:  # print every 100 mini-batches
                print('[epochs: {}, mini-batches: {}, images: {}] loss: {:.3f}'.format(
                    epoch + 1, itr + 1, (itr + 1) * batch_size, running_loss / display_interval))
                running_loss = 0.0

        model.eval()
        valid_true = []
        valid_pred = []
        for data in valid_loader:
            images, labels = data
            valid_true.extend(labels.tolist())
            images, labels = images.to(device), labels.to(device)  # for GPU

            with torch.no_grad():
                logits = model(images)

            _, predicted = torch.max(logits.data, 1)
            valid_pred.extend(predicted.tolist())

        train_acc = accuracy_score(train_true, train_pred)
        valid_acc = accuracy_score(valid_true, valid_pred)
        print('    epocs: {}, train acc.: {:.3f}, valid acc.: {:.3f}'.format(epoch + 1, train_acc, valid_acc))
        print()

        epoch_list.append(epoch + 1)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

    print('Finished Training')

    print('Save Network')
    torch.save(model.state_dict(), 'model.pth')

    df = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list,
                       'valid/accuracy': valid_acc_list})

    print('Save Training Log')
    df.to_csv('train.log', index=False)


if __name__ == '__main__':
    main()
