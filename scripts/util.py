# -*- coding: utf-8 -*-
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class FoodDataset(Dataset):
    def __init__(self, root, images_text, classes_text, transform):
        self.root = root
        self.transform = transform

        self.table = []
        with open(images_text) as f:
            line = f.readline()
            self.table.append(line.strip())
            while line:
                line = f.readline()
                if line != '':
                    self.table.append(line.strip())

        classes = []
        with open(classes_text) as f:
            line = f.readline()
            classes.append(line.strip())
            while line:
                line = f.readline()
                if line != '':
                    classes.append(line.strip())
        self.label_dict = {k: i for i, k in enumerate(classes)}

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        class_and_img = self.table[item]

        img = Image.open(os.path.join(self.root, '{}.jpg'.format(class_and_img))).convert('RGB')
        img = self.transform(img)

        class_name = class_and_img.split('/')[0]
        label = torch.tensor(self.label_dict[class_name])

        return img, label


def create_dataset(dataset_name):
    if dataset_name == 'mnist':
        ch_list = [1, 64, 128, 128, 10]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        train_set = datasets.MNIST(root='../data/mnist_root', train=True, download=True, transform=transform)
        valid_set = datasets.MNIST(root='../data/mnist_root', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        ch_list = [3, 64, 128, 128, 256, 256, 10]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_set = datasets.CIFAR10(root='../data/cifar10_root', train=True, download=True, transform=train_transform)

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        valid_set = datasets.CIFAR10(root='../data/cifar10_root', train=False, download=True, transform=valid_transform)

    elif dataset_name == 'stl10':
        ch_list = [3, 64, 128, 128, 256, 256, 512, 512, 10]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_set = datasets.STL10(root='../data/stl10_root', split='train', download=True, transform=train_transform)

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        valid_set = datasets.STL10(root='../data/stl10_root', split='test', download=True, transform=valid_transform)

    elif dataset_name == 'food101':
        ch_list = [3, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 101]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_set = FoodDataset(root='/mnt/hdd/sika/data/food-101/images',
                                images_text='/mnt/hdd/sika/data/food-101/meta/train.txt',
                                classes_text='/mnt/hdd/sika/data/food-101/meta/classes.txt',
                                transform=train_transform)

        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        valid_set = FoodDataset(root='/mnt/hdd/sika/data/food-101/images',
                                images_text='/mnt/hdd/sika/data/food-101/meta/test.txt',
                                classes_text='/mnt/hdd/sika/data/food-101/meta/classes.txt',
                                transform=valid_transform)

    return train_set, valid_set, ch_list