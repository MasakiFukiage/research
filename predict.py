#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time
import confusionMatrix

import numpy as np
from numpy.random import *
import six

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L

import net
import data
import param

#パラメータ設定
in_size = param.in_
n_units = param.unit
out_size = param.out
data_num = param.data

#データ読み込み
test_data = data.read_data()
test_data = np.array(test_data)
lstm = net.LSTM(in_size, out_size, n_units)

#正解データ作成
le = []
length = []
label = []
l = []
la = []

for i in range(data_num):
    for j in range(out_size):
        le.append(len(test_data[i][j]))
    length.append(le)
    le = []

for k in range(data_num):
    for i in range(out_size):
        for j in range(length[k][i]):
            l.append(i)
        la.append(l)
        l = []
    label.append(la)
    la = []

label = np.array(label)
ans = [0 for i in range(out_size)]
matrix = [0 for i in range(data_num * out_size)]
true = [0 for i in range(data_num * out_size)]

for j in range(data_num):
    for i in range(out_size):
        true[i + out_size * j] = i

for ev in range(data_num):
    o = ev + 1
    model_name = "./model/model" + str(o) + ".model"
    serializers.load_npz(model_name, lstm)
    lstm.reset_state()
    for l in range(out_size):
        for frame in range(len(test_data[ev][l])):
            x = chainer.Variable(np.asarray([test_data[ev][l][frame]]).astype(
                np.float32))
            t = chainer.Variable(np.asarray([label[ev][l][frame]]).astype(
                np.int32))
            loss, acc, y = lstm(x, t, train=False)
            y = np.array(y.data)
        matrix[l + ev * out_size] = y.argmax()
        ans = [0 for i in range(out_size)]
acuracy = 0
for i in range(data_num * out_size):
    if matrix[i] == true[i]:
        acuracy += 1
acuracy /= out_size * data_num
acuracy *= 100
print(acuracy)
confusionMatrix.print_cmx(true, matrix)
print("complete")
