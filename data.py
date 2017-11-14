#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time
import os

import numpy as np
from numpy.random import *
import six
import pandas as pd

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L

#データ読み込みプログラム

label = []
k = 0
load_data = []
length = []
subjects_data = []
empty = []

#パスの準備
dir = '/home/fuki/DATA/MPI/large/'
subset_name = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09',
                        '10', '11', '12', '13'],
                       dtype=object)
label = 0
frame = 0
file_end = '/output.csv'


def read_data():
    label_data = []
    for subjects in range(10):
        o = subjects + 1
        for subset in range(13):
            label_name = os.listdir(dir + str(o) + '/' + subset_name[subset] +
                                    '/')
            if subset != 12:
                for label in range(4):
                    file_name = dir + str(o) + '/' + subset_name[
                        subset
                    ] + '/' + label_name[label] + file_end
                    label_data.append(np.loadtxt(file_name, delimiter=","))
            else:
                for label in range(3):
                    file_name = dir + str(o) + '/' + subset_name[
                        subset
                    ] + '/' + label_name[label] + file_end
                    label_data.append(np.loadtxt(file_name, delimiter=","))
        load_data.append(label_data)
        label_data = []

        #シーケンスの長さを読み込む
    for subjects in range(10):
        for label in range(51):
            length.append(len(load_data[subjects][label]))
    max = np.max(length)

    #パディング
    # pad = np.array([['-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1']], dtype=np.double)

    # for subjects in range(10):
    #     for label in range(51):
    #         while len(load_data[subjects][label])!=max:
    #             load_data[subjects][label]=np.concatenate([load_data[subjects][label],pad],axis=0)
    #np.array(load_data)
    #load_data = load_data / 5
    return load_data

# #データ読み込みプログラム

# #パラメータ設定
# in_size = 17 #入力データの次元数
# n_units = 4  #隠れ層のユニット数
# out_size = 4 #分類クラス数
# data_num = 10 #学習データ数 (人数)

# label = []
# k = 0
# load_data = []
# length = []
# """データ読みこみ"""
# def read_data():
#     #学習データの準備
#     train_data = np.ndarray((data_num, in_size), dtype=np.float32)
#     subjects_data = []
#     label_data = []
#     empty = []
#     padding = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])

#     #パスの準備
#     dir = '/home/fuki/DATA/MPI/large/'
#     subset_name = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13'],dtype = object)
#     label_name = np.array(['agree_considered','agree_continue','agree_pure','agree_reluctant'])
#     label = 0
#     frame = 0
#     file_end = '/output.csv'
#     dim_num = 17
#     k=0

#     #データ読み込み ((まだPart1だけ))

#     num = 1
#     subjects = str(num)

#     for j in range (10):
#         for i in range(4):#ラベルごとのデータ読み込み
#             file_name = dir + subjects + '/'+subset_name[0] +'/'+ label_name[i] + file_end
#             if np.loadtxt(file_name).max != 0:
#                 label_data.append(np.loadtxt(file_name))
#             k=0
#         load_data.append(label_data)
#         label_data=[]
#         num=num+1
#         subjects = str(num)

#     for i in range(10):
#         for j in range(4):
#             length.append(len(load_data[i][j]))

#     l = np.array(length)
#     #パディング（穴埋め処理）

#     for i in range(10):
#         for j in range (4):
#             while 0 != (l.max()-len(load_data[i][j])):
#                 load_data[i][j]=np.concatenate([load_data[i][j],padding],axis=0)

#     print("Loading data Done...")
#     return load_data
