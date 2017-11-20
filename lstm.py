#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time
import random
from tqdm import trange
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
import matplotlib.pyplot as plt

#パラメータ設定
in_size = param.in_
n_units = param.unit
out_size = param.out
data_num = param.data
epoch = param.epoch
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

# lstm = L.Classifier(lstm)
# lstm.compute_accuracy = False
for param in lstm.params():
    data = param.data
    data[:] = np.random.uniform(-0.2, 0.2, data.shape)  #-0.2~0.2の乱数初期化

# Set up optimizer
optimizer = optimizers.Adam()
optimizer.setup(lstm)

#訓練を行うループ
display = 10  # 何回ごとに表示するか
total_loss = 0
accuracy = 0
total_accuracy = 0
train_losses = []
train_loss = []
train_accuracy = []
train_accuracies = []
average_acc = 0
average_loss = 0
num = 0
num2 = 0
n = 0
test_loss = []
test_losses = []
start_time = time.time()
losses = 0
random = [i for i in range(out_size)]

#leave one person
for eval in range(data_num):
    o = eval + 1
    print("Training ", str(o), " Start...")
    #epoch loop
    for seq in trange(epoch, desc='epoch'):
        #subjects loop
        for subjects in trange(data_num, desc='subjects'):
            if eval != subjects:
                r = np.random.permutation(random)
                #label loop
                for l in trange(out_size, desc='label'):
                    # 前の系列の影響がなくなるようにリセット
                    lstm.reset_state()
                    #frame loop
                    for frame in range(len(test_data[subjects][r[l]])):
                        #全フレームを順伝搬
                        x = chainer.Variable(np.asarray(
                            [test_data[subjects][r[l]][frame]]).astype(
                                np.float32))
                        t = chainer.Variable(np.asarray(
                            [label[subjects][r[l]][frame]]).astype(
                                np.int32))
                        #順伝搬
                        loss, acc, y = lstm(x, t)
                        total_loss += loss.data
                        losses += loss
                        accuracy = acc.data
                        num += 1
                        n += 1
                losses /= len(test_data[subjects][l])
                n = 0
                total_accuracy += accuracy
                num2 += 1
                lstm.zerograds()
                losses.backward()
                optimizer.update()
        #各エポックの平均損失率
        if num != 0:
            average_loss = total_loss / num
            train_loss.append(average_loss)
            num = 0
            total_loss = 0
        #各エポックの平均精度
        if num2 != 0:
            average_acc = total_accuracy / num2
            train_accuracy.append(average_acc)
            total_accuracy = 0
            num2 = 0
        if seq % display == 0:
            print("sequence:{}, loss:{}, acc:{}".format(seq, average_loss,
                                                        average_acc))
    #評価データごとの損失率
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print("Training", str(o), " done...")
    model_name = "./model/model" + str(o) + ".model"
    serializers.save_npz(model_name, lstm)
    print("Saving ", model_name, " done...")

interval = int(time.time() - start_time)
print("実行時間: {}sec".format(interval))

#lossグラフ作成
all_loss = []
all_acc = []
for i in range(out_size):
    all_loss += train_losses[i]
    all_acc += train_accuracies[i]
all_loss /= out_size
all_acc /= out_size

plt.plot(all_loss, label="train loss")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title("train_loss")
plt.xlabel("epoch")
plt.ylabel("mean_loss")
plt.savefig("./figure/fig_loss.png")
plt.show()
plt.clf()

plt.plot(all_acc, label="train_accuracy")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title("train_accuracy")
plt.xlabel("epoch")
plt.ylabel("mean_acc")
plt.savefig("./figure/fig_acc.png")
plt.show()
plt.clf()

print("complete")
