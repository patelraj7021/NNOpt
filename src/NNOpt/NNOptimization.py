# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:43:17 2024

@author: patel
"""

import time
import os
import random
# from scikit-learn import train_test_split

# TensorFlow outputs unnecessary log messages
# GPU is working fine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from multiprocessing import Pool



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


def train_model(layers_num, training_in, true_out, eps,
                opt_alg='RMSprop', loss_func='mse', batch_sz=64,
                vali_perc=0.3):
    
    # check inputs
    if type(vali_perc) is not float:
        raise TypeError('Validation percentage must be float.')
    if vali_perc > 1. or vali_perc == 0:
        raise ValueError('Validation data percentage must be >0 and <1.')
        
    # get num. of features
    if len(training_in.shape) == 1:
        num_features = 1
    elif len(training_in.shape) > 2:
        raise TypeError('Rank of training data cannot be higher than two.')
    else:
        num_features = training_in.shape[1]
    
    model_arch = init_model_arch(layers_num, num_features+4,
                                 'relu', 'linear', num_features)
    
    model_arch.compile(loss=loss_func, optimizer=opt_alg)
    fitting_results = model_arch.fit(training_in, true_out, 
                                 epochs=eps, batch_size=batch_sz,
                                 validation_split=vali_perc, verbose=0)
    min_val_cost = round(min(fitting_results.history['val_loss']), 3)
    print(f'{layers_num}-layer model trained; min(val_cost) = {min_val_cost}')
          
    return fitting_results, model_arch


#def 


def loop_thru_models(features_in, target_vals):
    
    # check inputs
    if not tf.is_tensor(features_in):
        features_in = tf.convert_to_tensor(features_in)
    if not tf.is_tensor(target_vals):
        target_vals = tf.convert_to_tensor(target_vals)
    if features_in.shape[0] != target_vals.shape[0]:
        raise ValueError('Rows of input must be equal to rows of output.')
        
    # run
    # X_training, X_testing, y_training, y_testing = train_test_split(
    #     features_in, true_out, train_size=train_perc)
    
    num_layers_init_guess = [1, 2, 32, 64, 96, 128]
    
    pool_input = []
    for i in num_layers_init_guess:
        pool_input.append((i, features_in, target_vals, 100))
       
    cost_comp = {}
    
    with Pool(6) as p:
        pool_results = p.starmap(train_model, pool_input)
        
    cost_comp = []
    for pool_result in pool_results:
        cost_comp.append(pool_result[0])
           
    return dict(zip(num_layers_init_guess, cost_comp))
    

