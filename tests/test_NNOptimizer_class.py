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
    (([(-5, 5), (-2, 2)], 1000, -2, 2, multivar_first_order_poly, 3), [2])
    ])
class TestInit:
    
    def test_num_features(self, data_params_in, exp_output):
        data_in = generate_test_data(*data_params_in)
        nnopt_inst = NNO.NNOptimizer(*data_in)
        assert nnopt_inst.num_features == exp_output[0]


# creates NNOptimizer instances
@pytest.fixture(params=[
                (3, 'Nadam', 1, ([(-5, 5)], 1000, -2, 2, 
                                 univar_third_order_poly, 4)), 
                (1, 'adam', 2, ([(-5, 5)], 1000, -2, 2, 
                                 univar_third_order_poly, 4)),
                (4, 'rmsprop', 4, ([(-5, 5), (-2, 2)], 1000, -2, 2, 
                                   multivar_first_order_poly, 3))
                ])  
def create_test_cases(request):
    all_test_params = request.param
    data_in = generate_test_data(*all_test_params[-1])
    nnopt_inst = NNO.NNOptimizer(*data_in)
    in_params = all_test_params[:-1]
    nnopt_inst.add_model_to_scan_list(*in_params)
    return in_params, nnopt_inst

@pytest.fixture
def train_test_cases(create_test_cases):
    in_params, nnopt_inst = create_test_cases
    nnopt_inst.train_scanning_models(10)
    return in_params, nnopt_inst


class TestAddModelToScanList:
    
    def test_num_layers(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        model_out = nnopt_inst.scanning_models[-1]
        # len(*.layers) returns output layer too, so minus 1 for that
        assert len(model_out.layers) - 1 == in_params[0] 

    def test_num_nodes(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        model_out = nnopt_inst.scanning_models[-1]
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['units'] == nnopt_inst.num_nodes
        
    def test_activation_func(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        model_out = nnopt_inst.scanning_models[-1]
        first_layer_config = model_out.get_layer(index=0).get_config()
        assert first_layer_config['activation'] == nnopt_inst.hidden_act
    
    def test_output_func(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        model_out = nnopt_inst.scanning_models[-1]
        output_layer_config = model_out.get_layer(index=-1).get_config()
        assert output_layer_config['activation'] == nnopt_inst.out_act
        
    def test_bagging(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        models_made = len(nnopt_inst.scanning_models)
        assert models_made == in_params[2]
        
    def test_optimizer(self, create_test_cases):
        in_params, nnopt_inst = create_test_cases
        model_out = nnopt_inst.scanning_models[-1]
        optimizer = model_out.get_compile_config()['optimizer']
        if not isinstance(optimizer, str):
            # to account for nadam legacy usage
            optimizer = optimizer['class_name']
        assert optimizer == in_params[1]
        

class TestTrainModels:
    
    def test_empty_scanning_list(self, train_test_cases):
        in_params, nnopt_inst = train_test_cases
        assert len(nnopt_inst.scanning_models) == 0
    
    def test_total_model_num(self, train_test_cases):
        in_params, nnopt_inst = train_test_cases
        assert len(nnopt_inst.trained_models) == in_params[2]
        

class TestPredict:
    
    def test_predictions(self, train_test_cases):
        in_params, nnopt_inst = train_test_cases
        
        assert True == True
        
        

if __name__=='__main__':

    pytest.main()