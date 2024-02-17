# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:43:17 2024

@author: patel
"""


from multiprocessing import Pool
import numpy as np
import time


def wait_rand_time(time_in):
    time.sleep(time_in)
    return time_in


if __name__ == '__main__':
    pool_input = [10, 14, 20, 12, 24, 2]
    t_s = time.perf_counter()
    with Pool(6) as p:    
        testing_output = p.map(wait_rand_time, pool_input)
    t_f = time.perf_counter() - t_s
    print(t_f)
