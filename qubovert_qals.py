from pyeda.inter import *
from pyeda.boolalg.expr import *
from pyeda.boolalg import picosat
#from dwave.system.samplers import DWaveSampler
#from dwave.system.composites import EmbeddingComposite
from copy import deepcopy
import itertools
import multiprocessing
from qubovert.sat import AND, OR, XOR, XNOR, NAND, NOT
from qubovert.utils import QUBOMatrix, qubo_to_matrix
from neal import SimulatedAnnealingSampler
from qubovert import PCBO, PUBO
from qubovert.sim import anneal_pubo
from itertools import combinations, product
import copy
import time
from collections import defaultdict
import dwave_networkx as dnx
import networkx as nx
#!/usr/bin/env python3
import time
import numpy as np
import datetime
import sys
import csv
from random import SystemRandom
from QA4QUBO import matrix, vector, solver, tsp
from QA4QUBO.colors import colors

random = SystemRandom()
np.set_printoptions(linewidth=np.inf,threshold=sys.maxsize)

def manhattan(a, b):
    dist = sum(abs(val1-val2) for val1, val2 in zip(a,b))
    if dist==1:
        return True
    return False

def manhattan_dist(a, b):
    print(a,b)
    dist=0
    for k in range(nvars):
        print("k ",k)
        l= [(i,j) for (i,j) in grid if a[i][j]==k][0]
        l1= [(i,j) for (i,j) in grid if b[i][j]==k][0]
        dist+=manhattan2(l,l1)
        print("dist ",dist)
    return dist

def manhattan2(a, b):
    print('a',a)
    print('b',b)
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def neighbours(cell):
    for c in product(*(range(n-1, n+2) for n in cell)):
        if c != cell and 0 <= c[0] < nrows and 0<=c[1]<ncols and manhattan2(cell,c)==1:
            yield c

def check_sat(s,nvars,ngates,nrows,ncols,x):
    print(s)
    ### condition2 
    cnf1 = PCBO()
    #for k in range(nvars):
    #    cnf1.add_constraint_AND(*[customOneHot(*[x[i,j,k] for i in range(nrows) for j in range(nrows)])])
    for k in range(nvars):
        for i1,j1 in list(itertools.combinations([(i,j) for i in range(nrows) for j in range(ncols)],2)):
            cnf1.add_constraint_NAND(x[i1[0],i1[1],k],x[j1[0],j1[1],k],lam=100)
        cnf1.add_constraint_OR(*[x[i,j,k] for i in range(nrows) for j in range(ncols)],lam=10)

  #  print("\ncnf after cond2",cnf1.items())
    ### condition3
    #for c in range(ncells):
    #    for (k,l) in itertools.combinations(range(nvars),2):
     #       cnf1.add_constraint_AND(NAND(x[grid[c][0],grid[c][1],k],x[grid[c][0],grid[c][1],l]))
    for c in grid:
        for i,j in list(itertools.combinations(range(nvars),2)):
            cnf1.add_constraint_NAND(x[c[0],c[1],i],x[c[0],c[1],j],lam=100)
        
  #  print("\ncnf after cond3",cnf1.items())
    ### condition1
    for g in range(ngates):
        print("g = ", g)
        print("s[g] = ",s[g])
        for c in range(len(grid)):
#     these 2 OR for 3-qubit gates       cnf1.add_constraint_OR(NOT(x[grid[c][0],grid[c][1],s[g][0]]),OR(*[x[c1[0],c1[1],s[g][2]] for c1 in (y for y in grid if manhattan(grid[c],y)==True)]),lam=100)
#            cnf1.add_constraint_OR(NOT(x[grid[c][0],grid[c][1],s[g][1]]),OR(*[x[c1[0],c1[1],s[g][2]] for c1 in (y for y in grid if manhattan(grid[c],y)==True)]),lam=100)
             cnf1.add_constraint_OR(NOT(x[grid[c][0],grid[c][1],s[g][0]]),OR(*[x[c1[0],c1[1],s[g][1]] for c1 in (y for y in grid if manhattan(grid[c],y)==True)]),lam=100)
    print("len cnf", len(cnf1))
    sampler = SimulatedAnnealingSampler()
 #   print("properties",sampler.properties)

    ifsat = False
    N=1
    indN=0
    while not ifsat and indN<N:
        print("ifsat", ifsat)
        Q=cnf1.to_qubo().Q

        quso = cnf1.to_quso()
        qub_h, qub_J = quso.h, quso.J
       # print("bruteforce", cnf1.solve_bruteforce())
     #   print("anneal pubo", anneal_pubo(cnf1, num_anneals=1))
        print("Q",Q)
        Qmatr = qubo_to_matrix(Q,symmetric=False)
        Qlist = list(Q.keys())
        max_i = 0
        for (i,j) in Qlist:
            max_i = max(i,j, max_i)
        max_i+=1
        print("max_i",max_i)
 #       qubo_sample11 = sampler.sample_qubo(cnf1.to_qubo().Q, num_reads=2000)

        qubo_sample11, f_star, r_time = solver.solve(d_min = 70, eta = 0.02, i_max = 200, k = 1000, lambda_zero = 3/2, n = max_i, 
        N = 10, N_max = 100, p_delta = 0.01, q = 0.9, 
        topology = 'pegasus', Q = Q, Qmatr = Qmatr, qub_h = qub_h, qub_J = qub_J, cnf = cnf1, log_DIR = "", sim = True)

     #   print("parameters",sampler.parameters)
        solution11 = cnf1.convert_solution(qubo_sample11)
        print("! is solution valid? ", cnf1.is_solution_valid(solution11))
        indN+=1
        ifsat=cnf1.is_solution_valid(solution11)

    return cnf1.is_solution_valid(solution11), solution11


def astar_search(start,goal):
    open_list=[{'node':start,'g':0,'f':0}]
    closed_list=[]

    while open_list:
        m = open_list.pop()
        d1=copy.deepcopy(m['node'])
        g=copy.deepcopy(m['g'])
        closed_list.append(m)
        for (i,j) in grid:
            d=copy.deepcopy(d1)
            print("cell ",(i,j))
            k = d[i][j]
            print("k=",k)
            if (k>-1):
                for i1,j1 in neighbours((i,j)):
                    print("neighbor ",(i1,j1))
                    print("k = ",k)
                    d=copy.deepcopy(d1)
                    print("d=",d)
                    d[i][j]=-1
                    k1 = d[i1][j1]
                    d[i1][j1]=-1
                    d[i1][j1]=k
                    d[i][j]=k1
                    if np.array_equal(d,goal)==True:
                        print("equal")
                        return g+1
                    else:
                        f=g+1+manhattan_dist(d,goal)
                        print("f ",f)
                        open_list.append({'node':d,'g':g+1,'f':f})
        open_list = copy.deepcopy(sorted(open_list, key=lambda d: d['f'], reverse=True))
        if not open_list:
            return g

#start_time = time.time()
    
#orig_stdout = sys.stdout
#f = open('dwave_qubovert_out_circ1_logical_vars.txt', 'w')
#sys.stdout = f

#circuit 1 (described in our paper Plan A)
#gates=[[0,2],[1,3],[3,4],[1,4],[0,3],[0,2]]
#dep_graph={2:[1], 3:[1,2], 4:[0,1], 5:[0,4]} # 2nd gate depends on the 1st gate ...


# circuit 2 (new one with smaller subcircuits (should return to n/2 after unsuccessful trial on 3n/4))
#gates = [[0,2],[2,1],[2,3],[2,4],[3,4],[4,3]]
#dep_graph={1:[0], 2:[1], 3:[2], 4:[2,3], 5:[3,4]}

#circuit 3
#gates=[[1,3,4],[2,4,1],[2,3,0],[2,3,0]]
#dep_graph={1:[0], 3:[2]} # 1st gate depends on the 0st gate ...

#circuit 4
#gates=[[1,3,4],[1,4,3],[1,2,0]]
#dep_graph={1:[0], 3:[2]} # 1st gate depends on the 0st gate ...


#orig_stdout = sys.stdout
#f = open('qals_qubovert_toffoli_sa.txt', 'w')
#sys.stdout = f

#toffoli
#gates=[[2,0],[1,0],[2,1],[1,0],[2,1]]
#dep_graph={2:[1], 3:[2], 4:[3]}
gates=[[2,0],[1,0],[2,1]]
dep_graph={2:[1]}



### for help
#P = OR('x','y','z','w','o','p')
#print(P)
#print(P.degree)
#Q1 = QUBOMatrix()
#_reduce_degree1(P,Q1,2,None,None)
#print(Q1)




ngates_all = len(gates)
nvars = 3

#grid
nrows=1
ncols=nvars
grid = [(i,j) for i in range(nrows) for j in range(ncols)]
nsubcirc=0
x = exprvars("x", (0,nrows),(0,ncols),(0,nvars)) # x_ijk = 1, if q_k is placed on (i,j) cell on the grid
print("X ",x)
optimal_placements = []
optimal_s= []
free_gates = list(range(ngates_all))
included = [0] * ngates_all
excluded = [0] * ngates_all
busy = [0] * ngates_all
while free_gates:
    print("while free gates")
    free_gates = [i for i in range(len(gates)) if busy[i]==0]
    print("while free gates persist")
    s = [gates[i] for i in free_gates if busy[i]==0]
    print("s ",s)
    print(s)
    SATres = check_sat(s,nvars,len(s),nrows,ncols,x)
    print("sat? ",SATres[0])
    included = [0] * ngates_all
    excluded = [0] * ngates_all
    if SATres[0]==False:
        print("!!!")
        fail = len(free_gates)
        print("fail ", fail)
        success = 0
        while (success-fail)>1 or (fail-success)>1:
            print("abs")
            free_gates = list(range(ngates_all))
            included = [0] * ngates_all
            print("while included",included)
            s=[]
            ngates = int(np.floor((success+fail)/2))
            print("ngates ", ngates)
            i=0 # current number of gates
            sat = False
            br=False
            while i<ngates and not br and sum(included)<nvars:
                print("...")
                while i<ngates and ngates>=1 and free_gates and not br:
                    print("i in while", i)
                    print("free gates ", free_gates) 
                    gate1 = [y for y in free_gates if excluded[y]==0]
                    if (gate1):
                        gate = gates[gate1[0]]
                    else:
                        print("br ",br)
                        br=True
                        break
                    print(gate)
                    ind = gate1[0]
                    print("ind ", ind)
                    if ind not in dep_graph:
                        s.append(gate)
                        free_gates.pop(free_gates.index(ind))
                        i+=1
                        print("added")
                        print("s ",s)
                        print("i ",i)
                        included[ind] = 1
                        print("included ", included)
                    else:
                        print("included ", included)
                        prec_gates_included = [included[prec_gate] and busy[prec_gate] for prec_gate in (y for y in dep_graph[ind])]
                        print("preceding gates included? ", prec_gates_included)
                        if 0 not in prec_gates_included:
                            s.append(gate)
                            i+=1
                            included[ind] = 1
                            print("included ", included)
                            print("added")
                            print("s ",s)
                            print("i ",i)
                            free_gates.pop(free_gates.index(ind))
                        else:
                            excluded[ind]=1
                    busy[ind]=included[ind]
                res = check_sat(s,nvars,len(s),nrows,ncols,x)
                sat = res[0]
                sol = res[1]
                if not free_gates and sat==True:
                    fail=int(np.floor((success+fail)/2))
                if sat==True:
                    print("SAT - added")
                    print("i= ",i)
               #     br=True
                    break
                else:
                    if len(s)>0:
                        gate=s[len(s)-1]
                        i1=gates.index(gate)
                        s.pop()
                        print("pop s", s)
                        i-=1
                        print("i-1=",i)
                        excluded[i1]=1
                        print("excluded=",excluded)
                        busy[i1]=0
                        print("busy=",busy)
                    else:
                        break
                
                
                
            if sat==True:
                success=int(np.floor((success+fail)/2))
                print("s ",s)
                print("fail",fail)
                print("success", success)
                for sg in s:
                    busy[gates.index(sg)]=1
                if(len(optimal_placements)):
                    optimal_placements[nsubcirc-1]=sol
                    optimal_s[nsubcirc-1]=s
                else:
                    optimal_placements.append(sol)
                    optimal_s.append(s)
                print(sol)
            else:
                print("unsat")
                fail=int(np.floor((success+fail)/2))
                print("fail",fail)
                print("success", success)
                    
            if not free_gates:
                print("break")
                break
    else:
        for sg in s:
            print("sg ",sg)
            ind = [i for i in range(len(gates)) if np.array_equal(gates[i],sg) == True and busy[i]==0][0]
            print(ind)
            busy[ind]=1
            free_gates.pop(free_gates.index(ind))
            print("free gates!", free_gates)
        optimal_placements.append(SATres[1])
        optimal_s.append(s)
        nsubcirc+=1

print("!PLACEMENTS!")
placements = [[[-1 for x in range(ncols)] for x in range(nrows)] for p in range(len(optimal_placements))]
for p in range(len(optimal_placements)):
    for (i,j) in grid:
        for (key,val) in optimal_placements[p].items():
            for k in range(nvars):
                if str(key)=='x['+str(i)+','+str(j)+','+str(k)+']' and val==1:
                    placements[p][i][j]=k
    print(placements[p],"\n")

swap_count=0
for i in range(len(placements)-1):
    swap_count+=astar_search(placements[i],placements[i+1])

print("PLACEMENTS:\n")
for p in range(len(placements)):
    print("subcircuit:",optimal_s[p])
    print(placements[p])
print("SWAP count:",swap_count)

#end_time = time.time()
#print("time ", end_time - start_time)
    
#sys.stdout = orig_stdout
#f.close()
