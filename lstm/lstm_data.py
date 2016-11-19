#/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

package_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(package_path)

from dataset_handler.token_processor import TokenProcessor

class LstmData(TokenProcessor):
    '''Lstm Data module '''
    def __init__(self, random_seed, unkreplace_cnt=1, unkreplace_prob=0.2):
        super(LstmData, self).__init__(random_seed, unkreplace_cnt, unkreplace_prob)
    

    def init_data(self):
        super(LstmData, self).init_data()
    


