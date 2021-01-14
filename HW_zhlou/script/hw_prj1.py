#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time    : 2020/12/18 15:14
# @Author   : 'Lou Zehua'
# @File    : hw_prj1.py
'''
    UCI数据集: http://archive.ics.uci.edu/ml/index.php
    任选一个数据集
    任选一种ML算法：逻辑回归、决策树、神经网络、SVM等
    源码+实验报告
    交给助教
    Deadline: 学期末考试前

'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

encoding = 'utf-8'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
iris_path = os.path.join(BASE_DIR, 'data/iris/bezdekIris.data')

col_name = {
    0: 'sepal length',
    1: 'sepal width',
    2: 'petal length',
    3: 'petal width',
    4: 'class',
}
iris_type = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica',
}
col_name_indicator = {v: k for k, v in col_name.items()}
iris_type_indicator = {v: k for k, v in iris_type.items()}


def cn_ind2type(ind):
    return col_name[ind]


def cn_type2ind(t):
    return col_name_indicator[t]


def it_ind2type(ind):
    return iris_type[ind]


def it_type2ind(t):
    return iris_type_indicator[t]


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s accuracy: %.5f' % (tip, np.mean(acc)))


def print_accuracy(clf, x_train, y_train, x_test, y_test):
    # print('training score: %.5f' % (clf.score(x_train, y_train)))
    # print('testing score: %.5f' % (clf.score(x_test, y_test)))
    show_accuracy(clf.predict(x_train), y_train, 'training data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')


def plot_decision_regions(X, y, show_axis=None, test_idx=None, resolution=0.02):
    show_axis = show_axis or [0, 1]
    show_axis_label = list(map(cn_ind2type, show_axis))
    if list(X.shape)[1] != 2:
        print('[X should be a 2-dim-matrix with the shape: (n, 2). '
              'X will pruned to be a 2-dim-matrix according to show_axis.]')
    # 定义颜色和标记符号，通过颜色列图表生成颜色示例图
    marker = ('*', 's', 'v', '^', 'o', 'x')
    colors = ('red', 'green', 'blue', 'cyan', 'gray', 'purple')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 可视化决策边界
    x1_min, x1_max = X[:, show_axis[0]].min() - 1, X[:, show_axis[0]].max() + 1
    x2_min, x2_max = X[:, show_axis[1]].min() - 1, X[:, show_axis[1]].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制所有的样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, show_axis[0]], y=X[y == cl, show_axis[1]], alpha=0.8,
                    c=cmap(idx), marker=marker[idx], s=73, label=cl)

    # 使用小圆圈高亮显示测试集的样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, show_axis[0]], X_test[:, show_axis[1]], c='', alpha=1.0, linewidth=1,
                    edgecolors='black', marker='o', s=135, label='test set')
        plt.title('svm classification', fontsize=19, color='b')
        plt.xlabel(show_axis_label[0] + '(standardized)', fontsize=15)
        plt.ylabel(show_axis_label[1] + '(standardized)', fontsize=15)
        plt.legend(loc=2, scatterpoints=2)
        plt.show()


if __name__ == '__main__':

    # local settings
    # show_axis = [0, 1]
    show_axis = [2, 3]
    PREVIEW_RAW_DATA_2D = True
    PREVIEW_PREDICT_DATA_2D = True

    # 1. data preprocess
    iris_data = pd.read_csv(iris_path, delimiter=',', header=None, converters={4: it_type2ind})
    X = iris_data[list(range(0, 4))].__array__()
    y = iris_data[4].__array__()

    # data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # 2. preview raw data. set 'PREVIEW_RAW_DATA_2D = False' to skip this step.
    if PREVIEW_RAW_DATA_2D:
        show_axis_label = list(map(cn_ind2type, show_axis))
        X = X[:, show_axis]
        y = y
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', marker='*')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g', marker='s')
        plt.scatter(X[y == 2, 0], X[y == 2, 1], color='b', marker='v')
        plt.title('the relationship between sepal and target classes')
        plt.xlabel(show_axis_label[0])
        plt.ylabel(show_axis_label[1])
        plt.show()

    # 3. train
    svm_clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.10, tol=1e-3)
    svm_clf.fit(X_train_std, y_train)

    # 4. show result
    print_accuracy(svm_clf, X_train_std, y_train, X_test_std, y_test)

    # 5. preview result. set 'PREVIEW_PREDICT_DATA_2D = False' to skip this step.
    if PREVIEW_PREDICT_DATA_2D:
        plt.figure(figsize=(12, 7))
        plot_decision_regions(X_combined_std, y_combined, show_axis,
                              test_idx=range(len(y_train), len(y_combined)))
