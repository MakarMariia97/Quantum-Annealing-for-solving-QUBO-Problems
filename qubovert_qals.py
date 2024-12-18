from pyeda.inter import *
from pyeda.boolalg.expr import *
import itertools
from qubovert.sat import OR, NOT
from qubovert.utils import qubo_to_matrix
from qubovert import PCBO
import numpy as np
from QA4QUBO import solver
from utils_qals import manhattan, astar_search

# function to consruct boolean function in CNF for the circuit
def check_sat(s,nvars,ngates,nrows,ncols,x):
    # condition2 (each qubit is assigned to only one cell of the grid)
    cnf1 = PCBO()
    for k in range(nvars):
        for i1,j1 in list(itertools.combinations([(i,j) for i in range(nrows) for j in range(ncols)],2)):
            cnf1.add_constraint_NAND(x[i1[0],i1[1],k],x[j1[0],j1[1],k],lam=100)
        cnf1.add_constraint_OR(*[x[i,j,k] for i in range(nrows) for j in range(ncols)],lam=10)

    # condition 3 (at most one qubit is assigned to each cellof the grid)
    for c in grid:
        for i,j in list(itertools.combinations(range(nvars),2)):
            cnf1.add_constraint_NAND(x[c[0],c[1],i],x[c[0],c[1],j],lam=100)
        
    # condition1 (interacting qubts of all gates are adjacent)
    for g in range(ngates):
        for c in range(len(grid)):
            cnf1.add_constraint_OR(NOT(x[grid[c][0],grid[c][1],s[g][0]]),OR(*[x[c1[0],c1[1],s[g][1]] for c1 in (y for y in grid if manhattan(grid[c],y)==1)]),lam=100)

    ifsat = False
    N=1 # number of repetitions (needed because of probabilistic nature of Quantum Annealing)
    indN=0
    while not ifsat and indN<N:
        Q=cnf1.to_qubo().Q # convert cnf to QUBO using qubovert package
        quso = cnf1.to_quso() # convert cnf to Ising model
        qub_h, qub_J = quso.h, quso.J
        Qmatr = qubo_to_matrix(Q,symmetric=False)
        Qlist = list(Q.keys())
        max_i = 0
        for (i,j) in Qlist: # find dimension of QUBO
            max_i = max(i,j, max_i)
        max_i+=1

        # solve QUBO problem
        qubo_sample11, f_star, r_time = solver.solve(d_min = 70, eta = 0.02, i_max = 100, k = 1000, lambda_zero = 3/2, n = max_i, 
        N = 10, N_max = 80, p_delta = 0.01, q = 0.9, Q = Q, Qmatr = Qmatr, qub_h = qub_h, qub_J = qub_J, cnf = cnf1, sim = True)

        # check solution for validity
        solution11 = cnf1.convert_solution(qubo_sample11)
        ifsat=cnf1.is_solution_valid(solution11)        
        
        indN+=1

    return ifsat, solution11


# Toffoli gate
gates=[[2,0],[1,0],[2,1],[1,0],[2,1]]
dep_graph={2:[1], 3:[2], 4:[3]} # gate dependecies meaning that the 2nd gate in 'gates' list depends on the 1st one (starting with 0 index), etc.

ngates_all = len(gates)
nvars = 3

# grid
nrows=2
ncols=2
grid = [(i,j) for i in range(nrows) for j in range(ncols)]

nsubcirc=0 # initial number of subcircuits
x = exprvars("x", (0,nrows),(0,ncols),(0,nvars)) # x_ijk = 1, if qubit q_k is placed on (i,j) cell in the grid
optimal_placements = [] # list of qubit placements presented as characteristic vector X
optimal_s= [] # list of subcicruits
free_gates = list(range(ngates_all)) # initially all gates do not belong to any subcircuit
included = [0] * ngates_all # indicate if some gate belongs to some subcircuit
excluded = [0] * ngates_all # indicate if some gate becomes free again
busy = [0] * ngates_all
while free_gates:
    free_gates = [i for i in range(len(gates)) if busy[i]==0]
    s = [gates[i] for i in free_gates if busy[i]==0]
    SATres = check_sat(s,nvars,len(s),nrows,ncols,x)

    included = [0] * ngates_all
    excluded = [0] * ngates_all
    if SATres[0]==False:
        fail = len(free_gates)
        success = 0
        while (success-fail)>1 or (fail-success)>1: # binary search of satisfiable subcircuits (start with 'ngates_all' and then decrease by 2 (if fails) or add several gates (if has success))
            free_gates = list(range(ngates_all))
            included = [0] * ngates_all
            s=[] # current list of subcircuits
            ngates = int(np.floor((success+fail)/2))
            i=0 # current number of gates
            sat = False
            br=False
            while i<ngates and not br and sum(included)<nvars: # if subcircuit of size ngates has not been constructed yet
                while i<ngates and ngates>=1 and free_gates and not br: # and there are free gates
                    gate1 = [y for y in free_gates if excluded[y]==0] # take the 1st free gate
                    if (gate1):
                        gate = gates[gate1[0]]
                    else:
                        br=True
                        break
                    ind = gate1[0]
                    if ind not in dep_graph: # take into account gate dependencies
                        s.append(gate)
                        free_gates.pop(free_gates.index(ind))
                        i+=1
                        included[ind] = 1
                    else:
                        prec_gates_included = [included[prec_gate] and busy[prec_gate] for prec_gate in (y for y in dep_graph[ind])]
                        if 0 not in prec_gates_included:
                            s.append(gate)
                            i+=1
                            included[ind] = 1
                            free_gates.pop(free_gates.index(ind))
                        else:
                            excluded[ind]=1
                    busy[ind]=included[ind]
                res = check_sat(s,nvars,len(s),nrows,ncols,x) # check the constructed subcircuit for satisfiability
                sat = res[0]
                sol = res[1]
                if not free_gates and sat==True:
                    fail=int(np.floor((success+fail)/2))
                if sat==True:
                    break
                else: # exclude the last gate
                    if len(s)>0:
                        gate=s[len(s)-1]
                        i1=gates.index(gate)
                        s.pop()
                        i-=1
                        excluded[i1]=1
                        busy[i1]=0
                    else:
                        break
                
            if sat==True: # add solution to list of placements
                success=int(np.floor((success+fail)/2))
                for sg in s:
                    busy[gates.index(sg)]=1
                if(len(optimal_placements)):
                    optimal_placements[nsubcirc-1]=sol
                    optimal_s[nsubcirc-1]=s
                else:
                    optimal_placements.append(sol)
                    optimal_s.append(s)
            else:
                fail=int(np.floor((success+fail)/2))
                    
            if not free_gates:
                break
    else:
        for sg in s:
            ind = [i for i in range(len(gates)) if np.array_equal(gates[i],sg) == True and busy[i]==0][0]
            busy[ind]=1
            free_gates.pop(free_gates.index(ind))
        optimal_placements.append(SATres[1])
        optimal_s.append(s)
        nsubcirc+=1

# convert characteristic vectors X (denoting qubit position) obtained for each optimal subcircuit into grid
placements = [[[-1 for x in range(ncols)] for x in range(nrows)] for p in range(len(optimal_placements))]
for p in range(len(optimal_placements)):
    for (i,j) in grid:
        for (key,val) in optimal_placements[p].items():
            for k in range(nvars):
                if str(key)=='x['+str(i)+','+str(j)+','+str(k)+']' and val==1:
                    placements[p][i][j]=k

# calculates SWAPs needed to 'connect' the qubit placements
swap_count=0
for i in range(len(placements)-1):
    swap_count+=astar_search(placements[i],placements[i+1],grid,nrows,ncols,nvars)

print("PLACEMENTS:\n")
for p in range(len(placements)):
    print("subcircuit:",optimal_s[p])
    print(placements[p])
print("SWAP count:",swap_count)
