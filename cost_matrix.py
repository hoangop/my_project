import math


M = 100_000_000

cost_matrix = [
    [0, 1, 2, 2, M],
    [1, 0, 1, 2, 2],
    [2, 1, 0, 1, 2],
    [2, 2, 1, 0, 1],
    [M, 2, 2, 1, 0]
]


# MST based tsp approximation
import networkx as nx

# This code implements the simple MST based shortcutting approach that would yield factor of 2
# approximation for metric TSPs.
def minimum_spanning_tree_tsp(n, cost_matrix):
    G = nx.Graph()
    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=cost_matrix[i][j])
    T = nx.minimum_spanning_tree(G)
    print(f'MST for your graph has the edges {T.edges}')
    mst_cost = 0
    mst_dict = {} # store mst as a dictionary
    for (i,j) in T.edges:
        mst_cost += cost_matrix[i][j]
        if i in mst_dict:
            mst_dict[i].append(j)
        else:
            mst_dict[i] = [j]
        if j in mst_dict:
            mst_dict[j].append(i)
        else:
            mst_dict[j] = [i]
    print(f'MST cost: {mst_cost}')
    print(mst_dict)
    # Let's form a tour with short cutting
    def traverse_mst(tour_so_far, cur_node):
        assert cur_node in mst_dict
        next_nodes = mst_dict[cur_node]
        for j in next_nodes:
            if j in tour_so_far:
                continue
            tour_so_far.append(j)
            traverse_mst(tour_so_far, j)
        return
    tour = [0]
    traverse_mst(tour, 0)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    return tour, tour_cost

# optimal TSP tour taken from our notes using MTZ encoding
from pulp import *

def mtz_encoding_tsp(n, cost_matrix):
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'
    # create our encoding variables
    binary_vars = [ # add a binary variable x_{ij} if i not = j else simply add None
        [ LpVariable(f'x_{i}_{j}', cat='Binary') if i != j else None for j in range(n)] 
        for i in range(n) ]
    # add time stamps for ranges 1 .. n (skip vertex 0 for timestamps)
    time_stamps = [LpVariable(f't_{j}', lowBound=0, upBound=n, cat='Continuous') for j in range(1, n)]
    # create the problem
    prob = LpProblem('TSP-MTZ', LpMinimize)
    # create add the objective function 
    objective_function = lpSum( [ lpSum([xij*cj if xij != None else 0 for (xij, cj) in zip(brow, crow) ])
                           for (brow, crow) in zip(binary_vars, cost_matrix)] )
    
    prob += objective_function 
    
    # add the degree constraints
    for i in range(n):
        # Exactly one leaving variable
        prob += lpSum([xj for xj in binary_vars[i] if xj != None]) == 1
        # Exactly one entering
        prob += lpSum([binary_vars[j][i] for j in range(n) if j != i]) == 1
    # add time stamp constraints
    for i in range(1,n):
        for j in range(1, n):
            if i == j: 
                continue
            xij = binary_vars[i][j]
            ti = time_stamps[i-1]
            tj = time_stamps[j -1]
            prob += tj >= ti + xij - (1-xij)*(n+1) # add the constraint
    # Done: solve the problem
    status = prob.solve(PULP_CBC_CMD(msg=False)) # turn off messages
    assert status == constants.LpStatusOptimal, f'Unexpected non-optimal status {status}'
    # Extract the tour
    tour = [0]
    tour_cost = 0
    while len(tour) < n:
        i = tour[-1]
        # find all indices j such that x_ij >= 0.999 
        sols = [j for (j, xij) in enumerate(binary_vars[i]) if xij != None and xij.varValue >= 0.999]
        assert len(sols) == 1, f'{sols}' # there better be just one such vertex or something has gone quite wrong
        j = sols[0] # extract the lone solutio 
        tour_cost = tour_cost + cost_matrix[i][j] # add to the tour cost
        tour.append(j) # append to the tour
        assert j != 0
    i = tour[-1]
    tour_cost = tour_cost + cost_matrix[i][0]
    return tour, tour_cost
#test

#test that exact answer is 10^6 times smaller than approximate answer.
# compute MST based approximation
tour, tour_cost = minimum_spanning_tree_tsp(5, cost_matrix)
print(f'MST approximation yields tour is {tour} with cost {tour_cost}')
# compute exact answer
opt_tour, opt_tour_cost = mtz_encoding_tsp(5, cost_matrix)
print(f'Optimal tour is {opt_tour} with cost {opt_tour_cost}')
# check that the fraction is 1million times apart.
assert tour_cost/opt_tour_cost >= 1E+06, 'The TSP + shortcutting tour must be at least 10^6 times costlier than optimum. In your case, the ratio is {tour_cost/opt_tour_cost}'
print('Test passed: 7 points')