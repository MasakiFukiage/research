#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
from numpy.random import *
import six
#import lstm

import chainer
from chainer import Variable
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L


#LSTMのネットワーク定義
class LSTM(chainer.Chain):
    def __init__(self, in_size, out_size, n_units, train=True):
        super(LSTM, self).__init__(
            l1=L.Linear(in_size, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, out_size), )

    def __call__(self, x, t, train=True):
        h0 = self.l1(x)
        h1 = self.l2(h0)
        y = self.l3(h1)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t), y
        #return F.softmax_cross_entropy(y,t), y

    def reset_state(self):
        self.l2.reset_state()
