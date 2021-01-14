#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time    : 2020/12/18 21:42
# @Author   : 'Lou Zehua'
# @File    : hw_prj2.py
'''
    数据集：http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html
    任务：1000个文档分成20类，五重交叉验证结果，不要使用网站上的代码
    源码+实验报告
    交给助教
    Deadline: 学期末考试前

'''

import numpy as np
import os
import random

import nltk

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

encoding = 'utf-8'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
news_groups_dir = os.path.join(BASE_DIR, 'data/mini_newsgroups/')

# hyperparameters
K_fold = 5


def preprocess(news_groups_dir):
    X = []
    y = []
    news_groups = os.listdir(news_groups_dir)
    for news_group in news_groups:
        news_group_path = os.path.join(news_groups_dir, news_group)
        news_names = os.listdir(news_group_path)
        for news_name in news_names:
            with open(os.path.join(news_group_path, news_name), 'rb') as f:
                news_raw = f.read().decode('ANSI', 'ignore')
            words = nltk.tokenize.word_tokenize(news_raw)
            X.append(words)
            y.append(news_group)
    return X, y


def main():
    # 预处理：读入文本，转换为词袋矩阵
    X, y = preprocess(news_groups_dir)
    # shuffle
    data_class_zip = list(zip(X, y))  # zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_zip)  # 将data_class_list乱序
    X, y = zip(*data_class_zip)
    # 转换为array
    x = []
    for i in range(len(X)):
        x.append(' '.join(X[i]))
    x = np.array(x)
    y = np.array(y)
    # process pipline
    nbc = Pipeline([
        ('vect', TfidfVectorizer(stop_words=['\u3000', '\x00'], ngram_range=(1, 2), max_df=0.5
        )),
        ('clf', MultinomialNB(alpha=0.2, fit_prior=True)),  # alpha: Additive (Laplace/Lidstone) smoothing parameter.
    ])
    # K-fold cross validation
    avg_acc = 0
    i = 1
    kf = KFold(n_splits=K_fold)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        nbc.fit(x_train, y_train)  # 训练我们的多项式模型贝叶斯分类器
        y_predict = nbc.predict(x_test)  # 在测试集上预测结果
        y_predict = np.array(y_predict)
        y_test = np.array(y_test)
        acc = np.mean(y_predict == y_test)
        print('accurate_{}: '.format(i), acc)
        avg_acc += acc
        i += 1
    avg_acc /= K_fold
    print('{}-fold C.V. avg_acc: '.format(K_fold), avg_acc)


if __name__ == '__main__':
    main()
