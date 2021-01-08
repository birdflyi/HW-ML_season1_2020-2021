#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time    : 2021/1/5 11:49
# @Author   : 'Lou Zehua'
# @File    : hw_prj3.py
'''
    数据集：MNIST手写识别数据集 http://yann.lecun.com/exdb/mnist/
    任务：识别字符
    源码+实验报告
    交给助教
    Deadline: 学期末考试前

'''
import os

import numpy as np
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

encoding = 'utf-8'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mnist_dir = os.path.join(BASE_DIR, 'data/mnist')

BATCH_SIZE = 256  # mini-Batch设定为256 * 28 * 28，全部数据为60000 * 28 * 28
EPOCHS = 30
DEVICE = torch.device("cpu:0")


def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        fb_data = f.read()
    offset = 0
    fmt_header = '>ii'
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    offset += struct.calcsize(fmt_header)
    labels = []
    fmt_label = '>B'
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return np.array(labels)


class MnistDataset(Dataset):
    def __init__(self, X, Y):
        super(MnistDataset, self).__init__()
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y).long()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.Y[idx]


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # m * init_channels * 28 * 28:
        #     m = BATCH_SIZE or (N mod BATCH_SIZE): number of samples in each batch ,
        #     init_channels = 1 for gray image,
        #     img_shape = 28 * 28;
        #     loops: N = 60000, n = ceil(N / m) times loop for each epoch.
        # in_channels = 1，out_channels = 8，kernel_size = 5, stride = 1,
        #     image_shape = 24 * 24  # valid padding: floor((image_shape[i] - kernel_size) / stride) + 1 for i in [0, 1]
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        # image_shape = 12 * 12
        self.pool1 = nn.MaxPool2d(2, 2)
        # in_channels = 8，out_channels = 16，kernel_size = 3, stride = 1, image_shape = 10 * 10
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        # in_channels = out_channels * row * column
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # m * 1 * 28 * 28 -> m * 8 * 24 * 24
        x = self.pool1(x)  # m * 8 * 24 * 24 -> m * 8 * 12 * 12

        x = torch.relu(self.conv2(x))  # m * 8 * 12 * 12 -> m * 16 * 10 * 10

        x = x.view(x.size(0), -1)  # m * 16 * 10 * 10 -> m * 1600
        x = torch.relu(self.fc1(x))  # m * 1600 -> m * 120
        x = self.fc2(x)  # m * 120 -> m * 10

        out = torch.log_softmax(x, dim=1)  # log(softmax(x))
        return out


def m_train(model, train_loader, optimizer, epoch, device=DEVICE):
    model.train()
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 概率最大的索引
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += target.size(0)
        if (batch_idx + 1) % 30 == 0 or batch_idx + 1 >= len(train_loader):
            print(
                'Epoch {:2d}: Process: [{:5d}/{:5d} ({:.1f}%)]\tAccuracy: [{:5d}/{:5d} ({:.2f}%)]\tMean Loss: {:.6f}'.format(
                    epoch, batch_idx * BATCH_SIZE + len(data), len(train_loader.dataset),
                    100.0 * (batch_idx * BATCH_SIZE + len(data)) / len(train_loader.dataset), train_correct,
                    train_total, 100.0 * train_correct / train_total, loss.item()))


def m_test(model, test_loader, device=DEVICE):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]  # 概率最大的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Accuracy: {:5d}/{:5d} ({:.2f}%)\tMean Loss: {:.4f}\n'.format(
        correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset), test_loss))


if __name__ == '__main__':
    # read data
    X_train = decode_idx3_ubyte(os.path.join(mnist_dir, 'train-images.idx3-ubyte'))
    Y_train = decode_idx1_ubyte(os.path.join(mnist_dir, 'train-labels.idx1-ubyte'))
    X_test = decode_idx3_ubyte(os.path.join(mnist_dir, 't10k-images.idx3-ubyte'))
    Y_test = decode_idx1_ubyte(os.path.join(mnist_dir, 't10k-labels.idx1-ubyte'))
    # format data
    train_dataset = MnistDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = MnistDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MNIST_CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    for epoch_idx in range(EPOCHS):
        m_train(model, train_dataloader, optimizer, epoch_idx + 1)
        m_test(model, test_dataloader)
