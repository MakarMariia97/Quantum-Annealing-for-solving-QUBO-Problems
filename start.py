#!/usr/local/bin/python3
import pandas as pd
import datetime
import time
from QA4QUBO import matrix, vector, solver, tsp
from QA4QUBO.colors import colors
from os import listdir, mkdir, system, name
from os.path import isfile, join, exists
import sys
import numpy as np
import csv

np.set_printoptions(threshold=sys.maxsize)

def main():    
    _Q=[[0, -0.2, 0, 0, 1],
        [0, 1.0, -0.5, 1.0, 0.5],
        [0, 0, -0.9, 0, 0.1],
        [0, 0, 0, 0.7, 0.3],
        [0.1, 0, 0, 0.7, 0.1]]
    nn = 5

    for numberofiters in range(1):
        z, f_star, r_time = solver.solve(d_min = 7, eta = 0.02, i_max = 200, k = 1000, lambda_zero = 3/2, n = nn, 
        N = 10, N_max = 100, p_delta = 0.1, q = 0.99,#0.2, 
        topology = 'pegasus', Q = _Q, log_DIR = "", sim = True)

        print(z, f_star)
if __name__ == '__main__':
    main()
