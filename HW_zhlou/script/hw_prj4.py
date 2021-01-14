#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time    : 2021/1/5 14:40
# @Author   : 'Lou Zehua'
# @File    : hw_prj4.py
'''
    Final Project: 人脸识别
    数据：CMU Machine Learning Faces
    http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html
    描述：20人，每人32张脸图(含表情)
    任务1：使用机器学习进行人脸分类识别，给出识别准确率
    任务2：使用聚类或分类算法发现表情相似的脸图
    源码+实验报告
    交给助教
    Deadline: 学期末考试前

'''

import numpy as np
import os
import re

import torch
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

encoding = 'utf-8'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_groups_dir = os.path.join(BASE_DIR, 'data/faces_4/')
y_person_dict = {k: v for k, v in enumerate(
    ['an2i', 'bpm', 'choon', 'karyadi', 'megak', 'phoebe', 'sz24', 'at33', 'ch4f', 'danieln', 'kawamura', 'mitchell',
     'saavik', 'tammo', 'boland', 'cheyer', 'glickman', 'kk49', 'night', 'steffi'])}
y_person_dict_reverse = {v: k for k, v in y_person_dict.items()}
y_facial_expression_dict = {k: v for k, v in enumerate(['angry', 'sad', 'happy', 'neutral'])}
y_facial_expression_dict_reverse = {v: k for k, v in y_facial_expression_dict.items()}

BATCH_SIZE = 20
EPOCHS = 10


def type2idx(idx, t_dict):
    return t_dict[idx]


def idx2type(t, t_dict):
    return t_dict[t]


# read pgm files, as the same as pyplot.imread(fname).
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(P5.*[\r\n]"
            b"(\d+)\s(\d+)[\r\n]"
            b"(\d+).*[\r\n])", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def preprocess(faces_groups_dir):
    X = []
    y_person = []
    y_facial_expression = []
    faces_groups = os.listdir(faces_groups_dir)
    for faces_group in faces_groups:
        faces_group_path = os.path.join(faces_groups_dir, faces_group)
        person_faces = os.listdir(faces_group_path)
        for person_face in person_faces:
            # image = pyplot.imread(os.path.join(faces_group_path, person_face))
            image = read_pgm(os.path.join(faces_group_path, person_face), '<')
            X.append(image)
            y_person.append(y_person_dict_reverse[faces_group])
            facial_expression = person_face.split('_')[2]
            y_facial_expression.append(y_facial_expression_dict_reverse[facial_expression])
    X = np.asarray(X, dtype=np.float32)
    # X = X / np.max(X)
    y_person = np.asanyarray(y_person, dtype=np.int32)
    y_facial_expression = np.asanyarray(y_facial_expression, dtype=np.int32)
    return X, y_person, y_facial_expression


class FaceDataset(Dataset):
    def __init__(self, X, Y):
        super(FaceDataset, self).__init__()
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y).long()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item].unsqueeze(0), self.Y[item]


class FaceVGG(nn.Module):
    def __init__(self):
        super(FaceVGG, self).__init__()
        self.features = self._make_layers([8, 8, 'M', 16, 16, 'M', 16, 16, 'M'])
        self.classifier = nn.Linear(192, 20)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x  # rotate
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def get_dataloader(X, y, test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    train_dataset = FaceDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = FaceDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader


# 人物分类
def face_recognition(X, y):
    train_dataloader, test_dataloader = get_dataloader(X, y, test_size=0.25)
    global model
    model = FaceVGG()
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # print(model.features)
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        # train
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()
            _, pre = outputs.max(1)
            train_correct += pre.eq(targets).sum().item()
            train_total += targets.size(0)

            if (batch_idx + 1) % 5 == 0 or batch_idx + 1 >= len(train_dataloader):
                print(
                    'Epoch {:2d}: Process: [{:3d}/{:3d} ({:.1f}%)]\tAccuracy: [{:3d}/{:3d} ({:.1f}%)]\tMean Loss: {:.5f}'.format(
                        epoch + 1, batch_idx * BATCH_SIZE + len(inputs), len(train_dataloader.dataset),
                        100.0 * (batch_idx * BATCH_SIZE + len(inputs)) / len(train_dataloader.dataset), train_correct,
                        train_total, 100.0 * train_correct / train_total, loss.item()))
        # test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, pre = outputs.max(1)
                test_correct += pre.eq(targets).sum().item()
                test_total += targets.size(0)
        test_loss /= len(test_dataloader.dataset)
        print(
            '\nTest Set: Epoch {:2d}: Accuracy: [{:3d}/{:3d} ({:.1f}%)]\tMean Loss: {:.5f}\n'.format(
                epoch + 1, test_correct, test_total, 100.0 * test_correct / test_total, test_loss))
        if test_correct >= test_total:
            print('Face Recognition training task breaks with an 100% accuracy!\n')
            break


def face_clustering(k_min, k_max):
    k_min = max(1, int(k_min))
    k_max = max(1, k_min, int(k_max))

    features = model(torch.Tensor(X).unsqueeze(1)).detach().numpy()
    SSE = []
    for c_i in range(k_min, k_max):
        clf = KMeans(n_clusters=c_i)
        clf.fit(features)
        SSE.append(clf.inertia_)
    pyplot.plot([i for i in range(k_min, k_max)], SSE)
    pyplot.xlabel('k(number of clusters)')
    pyplot.ylabel('Sum of Squared Error')
    pyplot.show()

    # k = 20 开始收敛
    clf = KMeans(n_clusters=20)
    clusters_predicted = clf.fit(features).predict(features)
    print("clusters_predicted: \n", clusters_predicted)


if __name__ == "__main__":
    X, y_person, y_facial_expression = preprocess(img_groups_dir)
    # # 查看数据
    # for x_i, y_i_p, y_i_fe in zip(X, y_person, y_facial_expression):
    #     pyplot.imshow(x_i)
    #     pyplot.show()
    #     print(x_i, y_i_p, y_i_fe)
    face_recognition(X, y_person)
    face_clustering(1, 40)
