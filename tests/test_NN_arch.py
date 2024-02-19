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
from src.temp_proj_name import NN_optimization



class TestNNArch:
    
    def test_int_0(self):
        assert pytest.raises(TypeError, NN_optimization.init_model_arch,
                             ('3', 8, 'relu', 'linear', 4))
        
    def test_int_1(self):
        assert pytest.raises(TypeError, NN_optimization.init_model_arch,
                             (3, 8., 'relu', 'linear', 4))

if __name__=='__main__':
    pytest.main()