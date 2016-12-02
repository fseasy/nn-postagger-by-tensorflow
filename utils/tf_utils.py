#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import sys
import logging

class TFUtils(object):
    optimizer_name2constructor = {
        "sgd": tf.train.GradientDescentOptimizer,
        "adadelta": tf.train.AdadeltaOptimizer,
        "adagrad": tf.train.AdagradOptimizer,
        "momentum": tf.train.MomentumOptimizer,
        "adam": tf.train.AdamOptimizer,
        "rmsprop": tf.train.RMSPropOptimizer
    }
    @classmethod
    def get_optimizer(cls, opt_name="sgd"):
        opt_name = opt_name.lower()
        try :
            return cls.optimizer_name2constructor[opt_name]
        except KeyError:
            logging.getLogger(__name__).error("optmizer '{}' has not added to list.".format(opt_name))
            raise 

def unit_test():
    logging.basicConfig(level=logging.DEBUG)
    adam = TFUtils.get_optimizer("adam")()
    try:
        not_valid = TFUtils.get_optimizer("not_valid")()
    except KeyError:
        print("raise KeyError ok.", file=sys.stderr)
    except:
        print("other exception raised.", file=sys.stderr)
        raise

if __name__ == "__main__":
    unit_test()
