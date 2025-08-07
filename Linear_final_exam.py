from pulp import *

def calculateOptimalPlan(n, edge_list, supplies, debug=False):

    assert n >= 1
    assert all(0 <= i < n and 0 <= j < n and i != j and c >= 0 for (i, j, c) in edge_list)
    assert len(supplies) == n

    prob = LpProblem("Optimal Gas Transportation", LpMinimize)

    # Create a set of all unique directed edges from the undirected edge list
    directed_edges = set()
    for i, j, _ in edge_list:
        directed_edges.add((i, j))
        directed_edges.add((j, i))

    # Decision Variables: flow_i_j represents flow from node i to node j
    flow_vars = LpVariable.dicts("flow",directed_edges,lowBound=0,cat='Continuous')

    # Map undirected edges to directed flow variables and their costs
    edge_costs = {}
    for i, j, c in edge_list:
        edge_costs[(i, j)] = c
        edge_costs[(j, i)] = c # Cost is the same in both directions

    # Objective Function: Minimize total transportation cost
    prob += lpSum(flow_vars[(i, j)] * edge_costs[(i, j)]
                      for (i, j) in directed_edges), "Total Transportation Cost" # Iterate over directed_edges

    # Constraints:
    for k in range(n):
        inflow = lpSum(flow_vars[(i, k)] for i, j_node in directed_edges if j_node == k)
        outflow = lpSum(flow_vars[(k, j)] for i_node, j in directed_edges if i_node == k)
        if supplies[k] < 0:
            prob += (inflow - outflow == -supplies[k]), f"Demand_Constraint_Node_{k}"
        else:
            prob += (outflow - inflow <= supplies[k]), f"Supply_Constraint_Node_{k}"

    # Solve the problem
    prob.solve()

    if debug:
        print("Status:", LpStatus[prob.status]) 
        print("Optimal Cost:", value(prob.objective)) 

    # Prepare the result dictionary
    optimal_plan = {}
    if prob.status == 1: 
        for (i, j) in flow_vars:
            if flow_vars[(i, j)].varValue is not None and flow_vars[(i, j)].varValue > 1e-9:
                optimal_plan[(i, j)] = flow_vars[(i, j)].varValue
    
    return optimal_plan



#test
def test_solution(n, edge_list, supplies, solution_map, expected_cost):
    cost = 0
    outflows = [0]*n
    inflows = [0]*n
    for (i,j,c) in edge_list:
        if (i,j) in solution_map: 
            flow = solution_map[(i,j)]
            cost += c * flow
            assert flow >= 0, f'flow on edge {(i,j)} is negative --> {flow}'
            outflows[i] += flow 
            inflows[j] += flow
        elif (j,i) in solution_map:
            flow = solution_map[(j,i)]
            cost += c * flow
            assert flow >= 0, f'flow on edge {(j,i)} in negative --> {flow}'
            outflows[j] += flow
            inflows[i] += flow 
    for (i, s) in enumerate(supplies):
        if s > 0:
            assert outflows[i]  - inflows[i] <= s, f'Vertex {i} constraint violated: total outflow = {outflows[i]} inflow = {inflows[i]}, supply = {s}'
        else:
            assert abs(inflows[i]-outflows[i] + s) <= 1E-2,f'Vertex{i} constraint violated: inflow = {inflows[i]} outflow={outflows[i]}, demand = {-s}'
    if expected_cost != None:
        assert abs(expected_cost - cost) <= 1E-02, f'Expected cost: {expected_cost}, your algorithm returned: {cost}'
    print('Test Passed!')

from random import randint,seed
def gen_random_test(n, num_edges):
    assert n >= 1
    edge_list = [(i,i+1, randint(2,10)) for i in range(n-1)]
    
    while len(edge_list) < num_edges:
        i = randint(0, n-1)
        j = randint(0, n-1)
        (i,j) = (min(i,j), max(i,j))
        if i == j: 
            continue 
        if any( ihat == i and jhat == j for (ihat, jhat, _) in edge_list):
            continue
        c = randint(2, 10)
        edge_list.append((i,j,c))
    tot = 0
    supplies=[]
    for i in range(n-1):
        si = randint(-100, 100) 
        supplies.append(si)
        tot = tot + si

    if tot <= 0:
        supplies.append(-tot)
    else:
        supplies.append(randint(1-tot, 0))
    
    return (n, edge_list, supplies)

seed(10001)
(n, edge_list,supplies) = gen_random_test(50, 100)
print(edge_list)
sol_map = calculateOptimalPlan(n, edge_list, supplies, debug=True)
test_solution(n, edge_list, supplies, sol_map,None)

(n, edge_list,supplies) = gen_random_test(45, 50)
print(edge_list)
sol_map = calculateOptimalPlan(n, edge_list, supplies, debug=True)
test_solution(n, edge_list, supplies, sol_map,None)


(n, edge_list,supplies) = gen_random_test(15,80)
print(edge_list)
sol_map = calculateOptimalPlan(n, edge_list, supplies, debug=True)
test_solution(n, edge_list, supplies, sol_map,None)

print('15 points!')