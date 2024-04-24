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



def third_order_poly(x, c):
    return c[3]*(x**3) + c[2]*(x**2) + c[1]*x + c[0]


def create_third_order_poly_test(x_start, x_end, num_x, 
                                 coeff_start, coeff_end):
    x_range = np.linspace(x_start, x_end, num_x)
    rand_coeffs = np.random.uniform(coeff_start, coeff_end, 4)
    y_poly = third_order_poly(x_range, rand_coeffs)
    noise_amp = abs((max(y_poly) - min(y_poly))*0.025)
    noise = np.random.normal(-noise_amp, noise_amp, len(x_range))
    return x_range, y_poly+noise


# working input cases
@pytest.fixture(params=[
                (create_third_order_poly_test(-5, 5, 1000, -2, -2))
                ])  
def create_NNOptimizer_instance(request):
    in_params = request.param
    output = NNO.NNOptimizer(*in_params)
    return in_params, output


class TestInit:
    
    
    def test_num_features(self, create_NNOptimizer_instance):
        in_params, output = create_NNOptimizer_instance
        X = in_params[0]
        if len(X.shape) == 1:
            X_num_cols = 1
        else:
            X_num_cols = X.shape[-1]
        assert output.num_features == X_num_cols
        
        

if __name__=='__main__':

    pytest.main()