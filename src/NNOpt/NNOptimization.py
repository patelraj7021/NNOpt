# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:43:17 2024

@author: patel
"""

import time
import os
import random
from pathos.multiprocessing import ProcessPool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from contextlib import nullcontext

# TensorFlow outputs unnecessary log messages
# GPU is working fine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'



class NNOptimizer:
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        strategy = tf.distribute.MirroredStrategy()
    
    def __init__(self, X, y, vali_per=0.3):
        
        # TO DO: generalize this for any computer
        self.num_CPU_cores = 6
        self.num_GPUs = 6
        self.each_GPU_mem = 1024
        if len(tf.config.list_physical_devices('GPU')) > 0:
            split_GPU(self.num_GPUs, self.each_GPU_mem)
            self.GPU_mode = True
        else: 
            self.GPU_mode = False
        
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.Series) and \
            not isinstance(X, pd.DataFrame):
                raise TypeError(
                'X input must be a NumPy array or pandas Series/Dataframe.')
        if not isinstance(y, np.ndarray) and not isinstance(y, pd.Series):
                raise TypeError(
                'y output must be a NumPy array or pandas Series.')
        
        X_train, X_vali, y_train, y_vali = train_test_split(X, y,
                                                            test_size=vali_per,
                                                            random_state=752)
        
        self.num_training_rows = len(y_train)
        self.num_vali_rows = len(y_vali)
        
        training_data = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train))
        vali_data = tf.data.Dataset.from_tensor_slices(
            (X_vali, y_vali))
        
        self.num_features = get_num_features(X)
        self.num_nodes = self.num_features + 4
        self.hidden_act = 'relu'
        self.out_act = 'linear'
        self.buffer_size = 256
        batch_size_guess = 256
        actual_batch_size = int(batch_size_guess/self.num_GPUs) \
                                * self.num_GPUs
        self.batch_size = actual_batch_size
        self.vali_perc = 0.3
        
        self.scanning_models = []
        self.trained_val_costs = {}
        self.trained_models = {}
        
        self.loss_func = 'mse'
        
        # AutoShard breaks everything
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        
        shuffled_training_data = training_data.shuffle(self.buffer_size)
        self.batched_training_data = shuffled_training_data.batch(
            self.batch_size, drop_remainder=True).with_options(options).\
            cache().prefetch(self.batch_size)
        
        self.batched_vali_data = vali_data.batch(
            self.batch_size, drop_remainder=True).with_options(options).\
            cache().prefetch(self.batch_size)
        
        
    def add_model_to_scan_list(self, lyrs, opt_alg, bag_num=1):
        if opt_alg.lower() == 'nadam':
            # need to use legacy nadam for distributed strategy
            opt_alg = keras.optimizers.legacy.Nadam()
        if self.GPU_mode:
            # strategy is only needed for tensorflow GPU case
            scope_obj = self.strategy.scope()
        else:
            scope_obj = nullcontext()
        with scope_obj:
            for i in range(bag_num):
                new_model = Sequential()
                new_model.add(keras.Input(shape=(self.num_features,),
                                          batch_size=self.batch_size))
                for i in range(lyrs):
                    new_model.add(Dense(self.num_nodes, 
                                        activation=self.hidden_act))
                new_model.add(Dense(1, activation=self.out_act))
                new_model.compile(loss=self.loss_func, optimizer=opt_alg)
                self.scanning_models.append(new_model)
        return
    
    
    def train_model(self, model):
        
        rand_ID = gen_rand_ID()
        
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.eps/10,
                                       mode='min',
                                       start_from_epoch=self.eps/10,
                                       restore_best_weights=True)
        
        check_point_path = os.path.join('model_saves', 
                                        f'weights_{rand_ID}.hdf5')
        save_model = ModelCheckpoint(check_point_path, 
                                     monitor='val_loss', 
                                     save_weights_only=True,
                                     save_best_only=True)
        
        fitting_results = model.fit(self.batched_training_data,
                                    epochs=self.eps,
                                    validation_data=\
                                        self.batched_vali_data,
                                    callbacks=[early_stopping,
                                               save_model],
                                    verbose=0)
        min_val_cost = round(min(fitting_results.history['val_loss']), 
                             3)
        
        model_num_layers = len(model.layers) - 1
        model_opt = model.get_compile_config()['optimizer']
        if not isinstance(model_opt, str):
            # to account for nadam legacy usage
            model_opt = model_opt['class_name']
        model_ID = f'{rand_ID}_{model_num_layers}_{model_opt}_{self.eps}'
        
        return model_ID, model, min_val_cost
    
    
    def train_scanning_models(self, eps):
        
        if len(self.scanning_models) == 0:
            raise RuntimeError('No models in scanning list.')
        
        self.eps = eps
        
        if self.GPU_mode:
            for model in self.scanning_models:     
                model_ID, model, min_val_cost = self.train_model(model)                
                self.trained_models[model_ID] = model
                self.trained_val_costs[model_ID] = min_val_cost
        else:
            with ProcessPool(self.num_CPU_cores) as p:
                training_out = p.map(self.train_model, self.scanning_models)
            for output in training_out:
                self.trained_models[output[0]] = output[1]
                self.trained_val_costs[output[0]] = output[2]
            
        self.scanning_models = []
        self.update_val_costs_df()
        
        return
    
    
    def update_val_costs_df(self):
        val_costs_df = pd.DataFrame.from_dict(self.trained_val_costs,
                                              orient='index',
                                              columns=['val_cost'])
        val_costs_df = val_costs_df.reset_index()
        val_costs_df[['ID', 'num_layers', 'optimizer', 'epochs']] = \
            val_costs_df['index'].str.split('_', expand=True)
        val_costs_df = val_costs_df.set_index('ID').drop(columns=['index'])
        self.val_costs_df = val_costs_df
        return
    
    
    def predict(self, pred_X):
        
        if len(self.final_models) == 0:
            raise RuntimeError('Optimized models not loaded in object.')
        if not isinstance(pred_X, np.ndarray) \
            and not isinstance(pred_X, pd.Series) \
            and not isinstance(pred_X, pd.DataFrame):
                raise TypeError(
                'X input must be a NumPy array or pandas Series/Dataframe.')
        
        pred_data = tf.data.Dataset.from_tensor_slices(pred_X)
        # AutoShard breaks everything
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        self.batched_pred_data = pred_data.batch(
            self.batch_size, drop_remainder=True).with_options(options).\
            cache().prefetch(self.batch_size)
        
        predictions = []
        
        for model in self.final_models:
            predictions.append(model.predict(self.batched_pred_data))
            
        bagged_pred = exp_bag_weighted_avg(self.final_model_val_costs,
                                           predictions)
            
        return bagged_pred
    
    
    
def split_GPU(num_splits, mem_each_split):
    # allow GPU memory to be fragmented so multiple models can be trained
    # at the same time
    
    config_input = []
    for i in range(num_splits):
        config_input.append(tf.config.LogicalDeviceConfiguration(
            memory_limit=mem_each_split))
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Create num_splits virtual GPUs with mem_each_split memory each
      try:
        tf.config.set_logical_device_configuration(
            gpus[0], config_input)
        # logical_gpus = tf.config.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
    return


def get_num_features(features_in):
    # get num. of features
    if len(features_in.shape) == 1:
        num_features = 1
    elif len(features_in.shape) > 2:
        raise TypeError('Rank of training data cannot be higher than two.')
    else:
        num_features = features_in.shape[1]
    return num_features


def gen_rand_ID():
    num_existing_models = len(os.listdir('model_saves'))
    rand_num = int(abs(7.*np.random.randn(1).item(0))*10000 \
                   - num_existing_models)
    rand_ID = str(rand_num) + str(num_existing_models)
    rand_ID = rand_ID.zfill(8)   
    return rand_ID


def exp_bag_weighted_avg(val_losses, predictions):
    # expects np arrays as input
    # val_losses: N by 1
    # predictions: len(rows of testing_X) by N
    # outputs np arrays
    # exp weighing makes neg. std dev predictions highly weighed (good preds.)
    # positive std dev predictions lowly weighed (bad preds.)
    # look at exp(-x) for intuition; normalizing centers val_losses at 0
    # implement maxes of -6 and +6 for normalized val_losses for stability
    
    val_losses_mean = np.mean(val_losses)
    val_losses_std = np.std(val_losses) 
    val_losses_norm = (val_losses - val_losses_mean) / val_losses_std
    val_losses_norm_clipped = np.clip(val_losses_norm, -6., 6.)
    weights = np.exp(-val_losses_norm_clipped)
    weighed_sum = np.dot(predictions, weights)
    weighed_average = weighed_sum / np.sum(weights)
    
    return weighed_average


def check_num_layers(features_in, target_vals, num_layers_list, eps, 
                     testing_X):
    
    num_CPU_cores = 6
    num_GPUs = 6
    each_GPU_mem = 1024
    if len(tf.config.list_physical_devices('GPU')) > 0:
        split_GPU(num_GPUs, each_GPU_mem)
        GPU_mode = True
    else: 
        GPU_mode = False
        
    bag_quantity = 3
    
    # make a folder to save model weights for ModelCheckpoint
    # need to do this before multiprocessing starts
    if not os.path.isdir('model_saves'):
        os.mkdir('model_saves')
    
    
    if not GPU_mode:
        # CPU mode
        pool_input = []
        for i in num_layers_list:
            pool_input.append((features_in, target_vals, bag_quantity, int(i), 
                               eps, testing_X))
        with Pool(num_CPU_cores) as p:
            pool_results = p.starmap(bag_models, pool_input)
    else:
        # GPU mode
        pool_results = []
        gpus = tf.config.list_logical_devices('GPU')
        # randomize list order so low nums aren't grouped together
        random.shuffle(num_layers_list)
        num_layers_array = np.array(num_layers_list)
        num_layers_array_chunks = np.array_split(num_layers_array, num_GPUs)
        for i, gpu in enumerate(gpus):
            with tf.device(gpu.name):
                pool_results.append(bag_models(features_in, target_vals, 
                                               bag_quantity, 
                                               list(num_layers_array_chunks[i])
                                               , eps, testing_X))
    
    # only want fitting results (history object) for this function
    cost_comp = []
    for pool_result in pool_results:
        cost_comp.append(pool_result[0])
           
    return dict(zip(num_layers_list, cost_comp))


def divide_chunks(list_in, len_chunk):       
    # looping till length l 
    for i in range(0, len(list_in), len_chunk):  
        yield list_in[i:i + len_chunk] 


def make_num_layers_list(start_num, end_num, test_num):
    float_array = np.logspace(np.log2(start_num), np.log2(end_num), 
                              num=test_num, base=2)
    int_array = float_array.astype(int)
    remove_dups_set = set(int_array)
    ordered_list = list(remove_dups_set)
    ordered_list.sort()
    
    return ordered_list


def gen_max_num_layers(num_features):
    # defined so that num_features = 1 -> max_num_layers = 128
    # num_features = 16 -> max_num_layers = 64
    b = -np.log(0.5) / 15
    a = 128 / np.exp(-b)
    # lowest should be 32
    max_num_layers = max(32, int(a*np.exp(-b*num_features))) 
    
    return max_num_layers


def find_best_num_layers(features_in, target_vals, testing_X):
    
    # check inputs
    if not tf.is_tensor(testing_X):
        testing_X = tf.convert_to_tensor(testing_X)
    if features_in.shape[0] != target_vals.shape[0]:
        raise ValueError('Num. of rows of features and targets must be equal.')
    
    num_features = get_num_features(features_in)
    
    broad_search_start = 2
    broad_search_end = gen_max_num_layers(num_features)
    broad_search_elements = 4
    broad_search_eps = 500
    broad_search_list = make_num_layers_list(broad_search_start, 
                                             broad_search_end,
                                             broad_search_elements)
    broad_search_results = check_num_layers(features_in, target_vals, 
                                            broad_search_list, 
                                            broad_search_eps,
                                            testing_X)
    
    return broad_search_results
    