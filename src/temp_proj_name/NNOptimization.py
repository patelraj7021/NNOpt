# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:43:17 2024

@author: patel
"""

import time
import os
import random
from sklearn import train_test_split

# TensorFlow outputs unnecessary log messages
# GPU is working fine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'



def wait_rand_time(time_in):
    time.sleep(time_in)
    return time_in


def init_model_arch(lyrs, nodes, hidden_act, out_act, num_in_vars):
    
    # check inputs
    if type(lyrs) is not int or type(nodes) is not int \
        or type(num_in_vars) is not int:
            raise TypeError(
            'Number of layers, nodes, and input variables must be integers.')
    if lyrs == 0 or nodes == 0 or num_in_vars == 0:
        raise ValueError(
        'Number of layers, nodes, and input variables must be greater than 0.')
    if type(hidden_act) is not str or type(out_act) is not str:
        raise TypeError(
        'Hidden and output layer activation names must be strings.')
    
    # run
    try:
        model_arch = Sequential()
        model_arch.add(Dense(nodes, input_shape=(num_in_vars,), 
                             activation=hidden_act))
        for i in range(lyrs-1):
            model_arch.add(Dense(nodes, activation=hidden_act))
        model_arch.add(Dense(1, activation=out_act))
    except ValueError:
        print(
        'Check keras documentation for valid activation and output functions.')
    return model_arch


def split_train_vali_test(features_in, true_out, train_perc, vali_perc):
    
    total_rows = true_out.shape[0]
    train_num = int(total_rows*train_perc)
    test_num = total_rows - train_num
    vali_num = int(train_num*vali_perc)
    
    train_indices = random.sample()
    
    return


def train_model(model_arch, features_in, true_out, 
                opt_alg='Nadam', loss_func='mse', 
                train_perc=0.8, vali_perc=0.3):
    
    # check inputs
    if not tf.is_tensor(features_in):
        features_in = tf.convert_to_tensor(features_in)
    if not tf.is_tensor(true_out):
        true_out = tf.convert_to_tensor(true_out)
    if features_in.shape[0] != true_out.shape[0]:
        raise ValueError('Rows of input must be equal to rows of output.')
    if type(train_perc) is not float or type(vali_perc) is not float:
        raise TypeError('Training & validation percentages must be floats.')
    if train_perc > 1. or train_perc == 0:
        raise ValueError('Training data percentage must be >0 and <1.')
    if vali_perc > 1. or vali_perc == 0:
        raise ValueError('Validation data percentage must be >0 and <1.')
        
    # run    
    X_training, X_testing, y_training, y_testing = train_test_split(
        features_in, true_out, train_size=train_perc)    
    
        
    model_out = model_arch   
    return model_out
    

