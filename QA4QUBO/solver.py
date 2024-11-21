#!/usr/bin/env python3
import time
import numpy as np
from QA4QUBO.matrix import generate_chimera, generate_pegasus
from QA4QUBO.script import annealer, hybrid
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
import datetime
import neal
import sys
import csv
import dimod
from random import SystemRandom
from QA4QUBO.colors import colors
from dwave.system.composites.embedding import EmbeddingComposite
random = SystemRandom()
np.set_printoptions(linewidth=np.inf,threshold=sys.maxsize)

def ising_qubo(z):
    x = []
  #  print("len ",len(z))
    for i in range(len(z)):
        x.append(int((1-z[i])/2))
    return x

def function_f(Q, z,n):
 #   print(z)
    x = ising_qubo(z)
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T)

def make_decision(probability):
    return random.random() < probability

def generate_theta(A):

    h = dict()
    J = dict()
    for row, col in A:
        h[row] = 1-2*random.random()
    for row, col in A:
        J[row, col] = 1-2*random.random()
      
    return h, J

    
def update_tabu_matr(h,J,S,lam):
    hnew = dict()
    Jnew = dict()
    n=len(h)
    for i in range(n):
        hnew[i]=h[i]+lam*S[i][i]
        if i<n-1:
            for j in range(i+1,n):
                if (i,j) in J:
                    Jnew[i,j] = J[i,j] + lam*S[i,j]
                else:
                    Jnew[i,j] = lam*S[i,j]
    return hnew, Jnew

def modify_temperature(p,p_delta,eta):
    return p-(p - p_delta)*eta

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, topology, Q, Qmatr, qub_h, qub_J, cnf, log_DIR, sim):
    
    try:
        if (not sim):
            sampler = EmbeddingComposite(DWaveSampler({'topology__type':'pegasus'}))
        else:
            sampler = neal.SimulatedAnnealingSampler()
        A = {(i,j) for i in range(n) for j in range(n)}
        #generate_pegasus(n)
        I = np.identity(n)
        p = 1.0
        h, J, ee = dimod.qubo_to_ising(Q)
        print("h, j", h, J)
        
        z_one = annealer(qub_h, qub_J, sampler, k, cnf)
        z_two = annealer(qub_h, qub_J, sampler, k, cnf)

        f_one = function_f(Qmatr, z_one,n).item()
        f_two = function_f(Qmatr, z_two,n).item()

        if (f_one < f_two):
            print("f star ", f_one)
            z_star = z_one
            f_star = f_one
        #    h_star, J_star = h, J,
            z_prime = z_two
        else:
            print("f star ", f_two)
            z_star = z_two
            f_star = f_two
            z_prime = z_one

        h_prime, J_prime = qub_h, qub_J
        
        if (f_one != f_two):
            S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
        else:
            S = np.zeros((n,n))
            

    except KeyboardInterrupt:
        exit()

    e = 0
    d = 0
    i = 1
    sum_time = 0
    lam = lambda_zero
    flag=0
    S0=np.zeros((n,n))
    while True:

        try:
            if (i % N == 0): # line 13 in the algorithm
                p = modify_temperature(p,p_delta,eta) # decrease the temperature
                
            if (make_decision(q)): # with probability q annealing is done
                h_prime, J_prime = update_tabu_matr(h_prime,J_prime,S,lam)
                z_prime = annealer(h_prime, J_prime, sampler, k, cnf)
            else: # with probability 1-q measurement is taken
              #  h_prime, J_prime = generate_theta(A)
                print("we are here")
                z_prime = [1 if random.random()<0.5 else -1 for i in range(len(z_prime))]
            
            if (z_prime != z_star):
                f_prime = function_f(Qmatr, z_prime,n).item() # line 18
                
                if (f_prime < f_star or ((f_prime >= f_star) and make_decision((p-p_delta)**(f_prime-f_star)))): # line 19: found a better solution or line 24: suboptimal acceptance
                    e,d = 0,0 # line 23
                    if(f_prime>=f_star):
                        d += 1
                    #else:
                    
                    z_prime, z_star = z_star, z_prime # line 21
                    f_star = f_prime
                  #  h_star, J_star = h_prime, J_prime # line 22

                S += np.outer(z_prime, z_prime) - I + np.diagflat(z_prime) # line 20: worse solution to tabu
                if (np.allclose(S,S0,atol=1e-2)):
                    flag = i
                lam = min(lambda_zero, (lambda_zero/(2+(i-1)-e+0.1)))
            else: # line 28: we found the same solution
                e += 1 

            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                break
            
            i += 1
        except KeyboardInterrupt:
            break
    if(flag>0):
        print("flag ", flag)
    return ising_qubo(np.atleast_2d(np.atleast_2d(z_star).T).T[0]), f_star, 0.0
