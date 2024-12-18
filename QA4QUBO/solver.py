import numpy as np
from QA4QUBO.script import annealer
from dwave.system.samplers import DWaveSampler
import neal
from random import SystemRandom
from dwave.system.composites.embedding import EmbeddingComposite
from utils_qals import check_validity

random = SystemRandom()

# function to convert spins to blloean variables
def ising_qubo(z):
    x = []
    for i in range(len(z)):
        x.append(int((1-z[i])/2))
    return x

# function to estimate cost function under current solution z
def function_f(Q, z):
    x = ising_qubo(z)
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T)

def make_decision(probability):
    return random.random() < probability

# update taby matrix S with 'bad' solution (corresponding to Ising model defined by h and J)
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

# function to decrease annealing temperature
def modify_temperature(p,p_delta,eta):
    return p-(p - p_delta)*eta

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, Q, Qmatr, qub_h, qub_J, cnf, sim):
    try:
        if (not sim):
            dwavesampler = DWaveSampler({'topology__type':'pegasus'})
            sampler = EmbeddingComposite(dwavesampler)
        else:
            sampler = neal.SimulatedAnnealingSampler()

        I = np.identity(n)
        p = 1.0
        
        # obtain two solutions
        z_one = annealer(qub_h, qub_J, sampler, k, cnf)
        z_two = annealer(qub_h, qub_J, sampler, k, cnf)

        f_one = function_f(Qmatr, z_one).item()
        f_two = function_f(Qmatr, z_two).item()

        # find which of these 2 is better 
        if (f_one < f_two):
            z_star = z_one
            f_star = f_one
            z_prime = z_two
        else:
            z_star = z_two
            f_star = f_two
            z_prime = z_one

        h_prime, J_prime = qub_h, qub_J
        
        # penalize worse solution
        if (f_one != f_two):
            S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
        else:
            S = np.zeros((n,n))

    except KeyboardInterrupt:
        exit()

    e = 0
    d = 0
    i = 1
    lam = lambda_zero
    ifsat=False
    while not ifsat:
        try:
            if (i % N == 0):
                p = modify_temperature(p,p_delta,eta) # decrease the temperature
            if (make_decision(q)): # with probability q annealing is done
                h_prime, J_prime = update_tabu_matr(h_prime,J_prime,S,lam)
                z_prime = annealer(h_prime, J_prime, sampler, k, cnf)
            else: # with probability 1-q measurement is taken
                z_prime = [1 if random.random()<0.5 else -1 for i in range(len(z_prime))]
            
            if (z_prime != z_star):
                f_prime = function_f(Qmatr, z_prime).item()
                if (f_prime < f_star or ((f_prime >= f_star) and make_decision((p-p_delta)**(f_prime-f_star)))): # line 19: found a better solution or line 24: suboptimal acceptance
                    e,d = 0,0 
                    if(f_prime>=f_star):
                        d += 1
                    z_prime, z_star = z_star, z_prime 
                    f_star = f_prime

                S += np.outer(z_prime, z_prime) - I + np.diagflat(z_prime) # add worse solution to tabu
                lam = min(lambda_zero, (lambda_zero/(2+(i-1)-e+0.1)))
                ifsat = check_validity(z_star, cnf)
            else: # we found the same solution
                e += 1 

            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                break
            
            i += 1
        except KeyboardInterrupt:
            break
    return ising_qubo(np.atleast_2d(np.atleast_2d(z_star).T).T[0]), f_star, 0.0
