#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
# not good
_cur_dir = os.path.dirname(__file__)
sys.path.append(_cur_dir)

from mlp_data import MlpData

def MlpNet(object):

    def build_net(self):
        self.graph = g = tf.Graph()
        with g.as_default(), tf.device("/cpu:0"):
            batch_x_input = tf.placfeholder()


