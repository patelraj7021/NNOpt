# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:43:17 2024

@author: patel
"""

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'



def wait_rand_time(time_in):
    time.sleep(time_in)
    return time_in


def init_model_arch(lyrs, nodes, hidden_act, out_act, num_in_vars):
    if type(lyrs) is not int or type(nodes) is not int \
        or type(num_in_vars) is not int:
            raise TypeError('Number of layers, nodes, and input variables \
                            must be integers.')
    
    model_arch = Sequential()
    model_arch.add(Dense(nodes, input_shape=(num_in_vars,), 
                         activation=hidden_act))
    for i in range(lyrs-1):
        model_arch.add(Dense(nodes, activation=hidden_act))
    model_arch.add(Dense(1, activation=out_act))
    return model_arch

