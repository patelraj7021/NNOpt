# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:37:34 2024

@author: patel
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.getcwd())) 
from src.NNOpt import NNOptimization as NNO
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'



def univar_third_order_poly(X, c):
    return c[3]*(X[:,0]**3) + c[2]*(X[:,0]**2) + c[1]*X[:,0] + c[0]


def multivar_first_order_poly(X, c):
    return c[0]*X[:,0] + c[1]*X[:,1] + c[2]


def generate_test_data(X_ranges, X_num,
                       coeff_start, coeff_end, func, num_coeff,
                       noise_amp_frac=0.025):
    
    ranges_list = []
    for active_range in X_ranges:
        new_range = np.linspace(active_range[0], active_range[1], X_num)
        ranges_list.append(new_range)
    X = np.array(ranges_list).T
    rand_coeffs = np.random.uniform(coeff_start, coeff_end, num_coeff)
    y = func(X, rand_coeffs)
    noise_amp = abs((max(y) - min(y))*noise_amp_frac)
    noise = np.random.normal(0, noise_amp, X_num)
    return X, y+noise


@pytest.mark.parametrize('data_params_in, exp_output', [
    (([(-5, 5)], 1000, -2, 2, univar_third_order_poly, 4), [1]),
    (([(-5, 5), (-2, 2)], 1000, -2, 2, multivar_first_order_poly, 4), [2])
    ])
class TestInit:
    
    def test_num_features(self, data_params_in, exp_output):
        data_in = generate_test_data(*data_params_in)
        nnopt_inst = NNO.NNOptimizer(*data_in)
        assert nnopt_inst.num_features == exp_output[0]
    
    def test_tensor_conversion(self, data_params_in, exp_output):
        data_in = generate_test_data(*data_params_in)
        nnopt_inst = NNO.NNOptimizer(*data_in)
        assert tf.is_tensor(nnopt_inst.features_in) == True


# cases for TestAddModelToScanList
@pytest.fixture(params=[
                (3, 'Nadam'), 
                (1, 'Nadam')
                ])  
def create_model_arch(request):
    data_in = generate_test_data([(-5, 5)], 1000, -2, 2, 
                                 univar_third_order_poly, 4)
    nnopt_inst = NNO.NNOptimizer(*data_in)
    in_params = request.param
    nnopt_inst.add_model_to_scan_list(*in_params)
    return in_params, nnopt_inst

class TestAddModelToScanList:
    
    def test_num_layers(self, create_model_arch):
        in_params, nnopt_inst = create_model_arch
        model_out = nnopt_inst.scanning_models[-1]
        # len(*.layers) returns output layer too, so minus 1 for that
        assert len(model_out.layers) - 1 == in_params[0] 

    def test_num_nodes(self, create_model_arch):
        in_params, nnopt_inst = create_model_arch
        model_out = nnopt_inst.scanning_models[-1]
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['units'] == nnopt_inst.num_nodes
        
    def test_num_inputs(self, create_model_arch):
        in_params, nnopt_inst = create_model_arch
        model_out = nnopt_inst.scanning_models[-1]
        input_layer_config = model_out.get_layer(index=0).get_build_config()
        assert input_layer_config['input_shape'] == (None, 
                                                     nnopt_inst.num_features)
        
    def test_activation_func(self, create_model_arch):
        in_params, nnopt_inst = create_model_arch
        model_out = nnopt_inst.scanning_models[-1]
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['activation'] == nnopt_inst.hidden_act
    
    def test_output_func(self, create_model_arch):
        in_params, nnopt_inst = create_model_arch
        model_out = nnopt_inst.scanning_models[-1]
        output_layer_config = model_out.get_layer(index=-1).get_config()
        assert output_layer_config['activation'] == nnopt_inst.out_act
        
        

if __name__=='__main__':

    pytest.main()