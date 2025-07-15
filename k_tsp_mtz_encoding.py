from pulp import *

def k_tsp_mtz_encoding(n, k, cost_matrix):
    # check inputs are OK
    assert 1 <= k < n
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'

    prob = LpProblem('kTSP', LpMinimize)

    # 1. Define Decision Variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), 0, 1, LpBinary)

    t = LpVariable.dicts("t", (i for i in range(1, n)), lowBound=1, upBound=n-1, cat='Integer')

    # 2. Objective Function
    prob += lpSum(cost_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # 3. Degree Constraints (as discussed in Problem A)
    prob += lpSum(x[(0, j)] for j in range(1, n)) == k, "Degree_Out_0"

    # k edges must enter vertex 0
    prob += lpSum(x[(i, 0)] for i in range(1, n)) == k, "Degree_In_0"

    for i in range(1, n):
        # Exactly one edge must enter vertex i
        prob += lpSum(x[(j, i)] for j in range(n) if j != i) == 1, f"Degree_In_{i}"
        # Exactly one edge must leave vertex i
        prob += lpSum(x[(i, j)] for j in range(n) if j != i) == 1, f"Degree_Out_{i}"

    # 4. Time Stamp Constraints (MTZ, as discussed in Problem B)
    M_val = n - 1 # or n is also common, but n-1 is sufficient for t_i in [1, n-1]

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # t_i - t_j + (n - 1) * x_i_j <= n - 2
                # Equivalent to t_j - t_i >= 1 if x_i_j = 1, otherwise loose
                prob += t[i] - t[j] + M_val * x[(i, j)] <= M_val - 1, f"MTZ_{i}_{j}"

    # 5. Solve the problem
    prob.solve()

    # 6. Extract Tours
    tours = [[] for _ in range(k)] # List to store k tours
    visited_edges = set() # To keep track of edges used in tours

    # Find starting edges from vertex 0 for each salesperson
    start_edges = []
    for j in range(1, n):
        if x[(0, j)].varValue > 0.5: # Check if edge (0, j) is selected
            start_edges.append((0, j))

    # Construct each tour
    for tour_idx in range(k):
        current_tour = [0] # Each tour starts at vertex 0
        current_node = 0
        
        # Find the next unvisited edge from 0
        for start_i, start_j in start_edges:
            if start_i == current_node and (start_i, start_j) not in visited_edges:
                current_node = start_j
                current_tour.append(current_node)
                visited_edges.add((start_i, start_j))
                break # Found the start for this salesperson

        # Traverse the rest of the tour
        while current_node != 0: # Continue until we return to vertex 0
            found_next = False
            for j in range(n):
                if j == current_node: # Avoid self-loops
                    continue
                if x[(current_node, j)].varValue > 0.5 and (current_node, j) not in visited_edges:
                    visited_edges.add((current_node, j))
                    current_node = j
                    if current_node != 0: # Don't add 0 again until the very end
                        current_tour.append(current_node)
                    found_next = True
                    break
            if not found_next and current_node != 0: # This means we are stuck, potentially a bug or problem
                print(f"Warning: Could not find next edge from {current_node}. Incomplete tour.")
                break
        

        if len(current_tour) > 1: # Ensure it's not just [0] for an empty salesperson route
            tours[tour_idx] = current_tour
        else: # Handle case where a salesperson might not have any cities assigned if k is too high
             tours[tour_idx] = [0] # An empty tour is just [0]

    # Let's re-extract the tours ensuring no double counting and matching exactly k tours.   
    final_tours = []
    current_x = {edge: x[edge].varValue for edge in x if x[edge].varValue > 0.5}
    
    for _ in range(k):
        tour = [0]
        # Find the starting edge for this tour
        start_node = 0
        next_node = -1
        
        # Iterate through possible next nodes from 0
        for j in range(1, n):
            if (start_node, j) in current_x:
                next_node = j
                del current_x[(start_node, j)] # Mark this edge as used
                break
        
        if next_node == -1: 
            final_tours.append([0])
            continue
            
        tour.append(next_node)
        current = next_node
        
        # Build the rest of the tour until it returns to 0
        while current != 0:
            found_next_edge = False
            for j in range(n):
                if j == current: continue # Skip self loop
                if (current, j) in current_x:
                    tour.append(j)
                    del current_x[(current, j)]
                    current = j
                    found_next_edge = True
                    break
            if not found_next_edge and current != 0:
                break # Break to avoid infinite loop in case of issue
        
        # Remove the final 0 if it was appended (as it's the start node implicitly)
        if tour[-1] == 0 and len(tour) > 1: # Remove if it's the end 0 and not just [0]
             tour = tour[:-1]
        
        final_tours.append(tour)

    final_tours.sort(key=lambda t: t[1] if len(t) > 1 else -1) # -1 for [0] tours, puts them last

    return final_tours


#test
from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(5):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

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
print('test passed: 15 points')