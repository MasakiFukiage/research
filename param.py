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

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L

#パラメータ設定
in_ = 23  #入力データの次元数
unit = 23  #中間層のユニット数
out = 51  #分類クラス数
data = 10  #学習データ数 (人数)
epoch = 100  #epoch数
