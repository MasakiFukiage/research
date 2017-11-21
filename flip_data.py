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

dir = '/home/xserve0/users/fukiage/DATA/MPI/large/'

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

    return load_data
