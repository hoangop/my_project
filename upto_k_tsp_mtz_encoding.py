from pulp import *

def upto_k_tsp_mtz_encoding(n, k, cost_matrix):
    # check inputs are OK
    assert 1 <= k < n
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'

    prob = LpProblem('kTSP_Upto', LpMinimize)

    # 1. Define Decision Variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), 0, 1, LpBinary)
    t = LpVariable.dicts("t", (i for i in range(1, n)), lowBound=1, upBound=n-1, cat='Integer')
    num_active_salespeople = LpVariable("num_active_salespeople", lowBound=1, upBound=k, cat='Integer')

    # 2. Objective Function
    prob += lpSum(cost_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # 3. Degree Constraints
    prob += lpSum(x[(0, j)] for j in range(1, n)) == num_active_salespeople, "Degree_Out_0"
    prob += lpSum(x[(i, 0)] for i in range(1, n)) == num_active_salespeople, "Degree_In_0"

    for i in range(1, n):
        prob += lpSum(x[(j, i)] for j in range(n) if j != i) == 1, f"Degree_In_{i}"
        prob += lpSum(x[(i, j)] for j in range(n) if j != i) == 1, f"Degree_Out_{i}"

    # 4. Time Stamp Constraints (MTZ)
    M_val = n - 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += t[i] - t[j] + M_val * x[(i, j)] <= M_val - 1, f"MTZ_{i}_{j}"

    # 5. Solve the problem
    prob.solve()

    # 6. Extract Tours from the Solution
    
    active_edges = {}
    for i in range(n):
        for j in range(n):
            if i != j and x[(i, j)].varValue > 0.5:
                active_edges[(i, j)] = True

    start_edges = []
    for j in range(1, n):
        if (0, j) in active_edges:
            start_edges.append((0, j))
            
    start_edges.sort(key=lambda edge: edge[1])

    extracted_tours = []

    for start_i, start_j in start_edges:
        tour = [0] 
        current_node = start_j
        tour.append(current_node)
        
        del active_edges[(start_i, start_j)]

        # Traverse the rest of the tour until it returns to vertex 0.
        while current_node != 0:
            found_next_edge = False
            for j in range(n):
                if j == current_node: # Skip self-loops
                    continue
                if (current_node, j) in active_edges:
                    if j == 0: 
                        del active_edges[(current_node, j)] 
                        current_node = j #
                        found_next_edge = True
                        break
                    else:
                        tour.append(j)
                        del active_edges[(current_node, j)]
                        current_node = j
                        found_next_edge = True
                        break
            if not found_next_edge and current_node != 0:
                break 
        
        # Add the tour to the list of extracted tours if it's not just [0]
        if len(tour) > 1:
            extracted_tours.append(tour)

    final_tours = sorted(extracted_tours, key=lambda t: t[1] if len(t) > 1 else float('inf'))

    return final_tours

#Test

from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) <= k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]

    print(f'Tour cost obtained by your code: {tour_cost}')
    #assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
    for i in range(1, n):
        is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
        assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'
    print('------')
print('test passed: 4 points')