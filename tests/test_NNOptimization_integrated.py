# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 01:19:13 2024

@author: patel
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.getcwd())) 
from src.NNOpt import NNOptimization as NNO
import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def third_order_poly(x, c):
    return c[3]*(x**3) + c[2]*(x**2) + c[1]*x + c[0]


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
        
        

if __name__=='__main__':

    X, y = create_third_order_poly_test(-5, 5, 2000, -2, 2)
    
    nnopt_inst = NNO.NNOptimizer(X, y, force_CPU=True)    
    nnopt_inst.find_best_opt_alg(eps=10)
    nnopt_inst.opt_alg_val_costs_df.to_csv('df_test_output.csv')
    # print(nnopt_inst.GPU_mode)
        
    
    
    
    
    
    # pytest.main()