from QA4QUBO import solver
from utils_qals import check_validity

def annealer(h, J, sampler, k, cnf):
    ifsat = False
    N=1 # repeat sampling several times (because of the probabilistic nature of annealing)
    indN=0
    while (not ifsat) and indN<N:
        response = sampler.sample_ising(h, J, num_reads=k, chain_strength = 10) 
        sol = response.first.sample.values()
        qubo_sol = solver.ising_qubo(list(sol)) # convert spins to boolean variables
        ifsat = check_validity(qubo_sol,cnf) # check if cnf satisfiable
        indN+=1

    return list(sol)
    
