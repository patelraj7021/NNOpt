# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 01:19:13 2024

@author: patel
"""

import sys
import os
import pytest
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.getcwd())) 
from src.NNOpt import NNOptimization as NNO
import keras
import numpy as np
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def third_order_poly(x, c):
    return c[3]*(x**3) + c[2]*(x**2) + c[1]*x + c[0]


# working input cases
@pytest.fixture(params=[
                (3, 8, 'relu', 'linear', 4), 
                (1, 4, 'relu', 'linear', 10)
                ])  
def create_model_arch(request):
    in_params = request.param
    model_out = NNO.init_model_arch(*in_params)
    return in_params, model_out


# @pytest.fixture(params=[
#                 (-5, 5, 1000, -2, 2, -1, 1)])
def create_third_order_poly_test(x_start, x_end, num_x, 
                                 coeff_start, coeff_end):
    # x_start, x_end, num_x, coeff_start, coeff_end, \
    #     noise_start, noise_end = request.param
    x_range = np.linspace(x_start, x_end, num_x)
    rand_coeffs = np.random.uniform(coeff_start, coeff_end, 4)
    y_poly = third_order_poly(x_range, rand_coeffs)
    noise_amp = abs((max(y_poly) - min(y_poly))*0.025)
    noise = np.random.normal(-noise_amp, noise_amp, len(x_range))
    return x_range, y_poly+noise


# test the init_model_arch function
class TestNNArch:
    
    # error input cases
    @pytest.mark.parametrize('in_params, exp_output', [
                                (('3', 8, 'relu', 'linear', 4), TypeError),
                                ((3, 8., 'relu', 'linear', 4), TypeError),
                                ((3, 8, 'relu', 'linear', [4]), TypeError),
                                ((0, 8, 'relu', 'linear', 4), ValueError),
                                ((3, 0, 'relu', 'linear', 4), ValueError),
                                ((3, 8, 'relu', 'linear', 0), ValueError),
                                ((3, 8, 'notreal', 'linear', 0), ValueError),
                                ((3, 8, 'relu', 'notreal', 0), ValueError),
                                ((3, 8, 1, 'linear', 4), TypeError),
                                ((3, 8, 'relu', 24, 4), TypeError)
                                # check for invalid activation inputs
                            ])
    def test_input_errors(self, in_params, exp_output):
        assert pytest.raises(exp_output, NNO.init_model_arch, *in_params)
    
    
    # rest of these uses the working input cases
    def test_num_layers(self, create_model_arch):
        in_params, model_out = create_model_arch
        # len(*.layers) returns output layer too, so minus 1 for that
        assert len(model_out.layers) - 1 == in_params[0] 

    def test_num_nodes(self, create_model_arch):
        in_params, model_out = create_model_arch
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['units'] == in_params[1]
        
    def test_num_inputs(self, create_model_arch):
        in_params, model_out = create_model_arch
        input_layer_config = model_out.get_layer(index=0).get_build_config()
        assert input_layer_config['input_shape'] == (None, in_params[4])
        
    def test_activation_func(self, create_model_arch):
        in_params, model_out = create_model_arch
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['activation'] == in_params[2]
    
    def test_output_func(self, create_model_arch):
        in_params, model_out = create_model_arch
        output_layer_config = model_out.get_layer(index=-1).get_config()
        assert output_layer_config['activation'] == in_params[3]
        
    
# test the model training loop
# class TestNNTraining:
    
#     def test_
        

if __name__=='__main__':

    X, y = create_third_order_poly_test(-5, 5, 1000, -2, 2)
    
    # X_tens = tf.convert_to_tensor(y)
    # print(X_tens.shape[0])
    # coarse_loop_results = NNO.loop_thru_models(X, y)
    plot_test = NNO.train_model(2, X, y, 500)
    plt.scatter(X, y, color='black')
    plt.scatter(X, plot_test.predict(X), color='red')
    # pytest.main()