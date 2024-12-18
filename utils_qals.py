from itertools import product
import copy
import numpy as np

# function to 'connect' qubit placements for subcircuits the circuit is divided into
def astar_search(start,goal,grid,nrows,ncols,nvars):
    open_list=[{'node':start,'g':0,'f':0}]
    closed_list=[]

    while open_list:
        m = open_list.pop()
        d1=copy.deepcopy(m['node'])
        g=copy.deepcopy(m['g'])
        closed_list.append(m)
        for (i,j) in grid:
            d=copy.deepcopy(d1)
            k = d[i][j]
            if (k>-1):
                for i1,j1 in neighbours((i,j),nrows,ncols): # swap position of qubit q_k to neighboring one
                    d=copy.deepcopy(d1)
                    d[i][j]=-1
                    k1 = d[i1][j1]
                    d[i1][j1]=-1
                    d[i1][j1]=k
                    d[i][j]=k1
                    if np.array_equal(d,goal)==True:
                        return g+1
                    else:
                        f=g+1+manhattan_dist(d,goal,grid,nvars)
                        open_list.append({'node':d,'g':g+1,'f':f})
        open_list = copy.deepcopy(sorted(open_list, key=lambda d: d['f'], reverse=True))
        if not open_list:
            return g

# function to calculate the Manhattan distance between all the qubits in 2 placements p1 and p2
def manhattan_dist(p1, p2,grid,nvars):
    dist=0
    for k in range(nvars):
        l= [(i,j) for (i,j) in grid if p1[i][j]==k][0]
        l1= [(i,j) for (i,j) in grid if p2[i][j]==k][0]
        dist+=manhattan(l,l1)
    return dist

# function to calculate the Manhattan distance between nodes a and b
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

# function to find neighbours of cell in grid
def neighbours(cell,nrows,ncols):
    for c in product(*(range(n-1, n+2) for n in cell)):
        if c != cell and 0 <= c[0] < nrows and 0<=c[1]<ncols and manhattan(cell,c)==1:
            yield c

# function to check cnf for satisfiability
def check_validity(z,cnf):
    sol = cnf.convert_solution(z)
    return cnf.is_solution_valid(sol)
