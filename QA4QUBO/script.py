import numpy as np
import time
import datetime
import dwave.inspector as inspector
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

def annealer(h, J, sampler, k, time=False):
    response = sampler.sample_ising(h, J, num_reads=k) 
    
    return list(response.first.sample.values())

def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())
    
