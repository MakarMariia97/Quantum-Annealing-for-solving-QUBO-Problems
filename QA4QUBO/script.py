import numpy as np
import time
import datetime
import dwave.inspector as inspector
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
from QA4QUBO import solver

def annealer(h, J, sampler, k, cnf, time=False):
    ifsat = False
    N=10
    indN=0
    while (not ifsat) and indN<N:
  #      print(indN)
     #   qpu = DWaveSampler()
      #  long_time = qpu.properties["annealing_time_range"][1]
        response = sampler.sample_ising(h, J, num_reads=k, chain_strength = 10) 
        sol = response.first.sample.values()
        qubo_sol = solver.ising_qubo(list(sol))
        cnf_sol = cnf.convert_solution(qubo_sol)
        ifsat = cnf.is_solution_valid(cnf_sol)
  #      print(sol)
        if ifsat:
            print("solution valid")
        indN+=1
    
    return list(sol)

def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())
    
