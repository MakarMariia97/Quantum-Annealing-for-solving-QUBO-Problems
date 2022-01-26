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
    _Q=[[0, -0.2, 0, 0],
        [0, 1.0, -0.5, 1.0],
        [0, 0, -0.9, 0],
        [0, 0, 0, 0.7]]
    nn = 4

    for numberofiters in range(10):
        z, r_time = solver.solve(d_min = 70, eta = 0.2, i_max = 200, k = 1, lambda_zero = 3/2, n = nn, 
        N = 10, N_max = 100, p_delta = 0.01, q = 0.99,#0.2, 
        topology = 'pegasus', Q = _Q, log_DIR = "", sim = True)

    min_z = solver.function_f(_Q,z).item()
    print("\t\t\t"+colors.BOLD+colors.OKGREEN+"RESULTS"+colors.ENDC+"\n")
    print(log_write("Z",z)+log_write("fQ",round(min_z,2)))

if __name__ == '__main__':
    main()
