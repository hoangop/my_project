import pulp

def tsp_with_extra_constraints(n, cost_matrix, constraints):
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'
    assert all( 1 <= i < n and 1 <= j < n and i != j for (i,j) in constraints)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", (i for i in range(n)), 0, n - 1, pulp.LpInteger)

    # Constraints
    prob = pulp.LpProblem("TSP_with_Extra_Constraints", pulp.LpMinimize)

    # 1. Each city must be entered exactly once
    for i in range(n):
        prob += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1, f"Enter_{i}"

    # 2. Each city must be exited exactly once
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1, f"Exit_{i}"

    # 3. MTZ Subtour Elimination Constraints (for i != 0, j != 0)
    prob += u[0] == 0, "U_0_fixed" 
    
    for i in range(n):
        for j in range(n):
            if i != j and i != 0 and j != 0: 
                prob += u[i] - u[j] + n * x[i, j] <= n - 1, f"MTZ_{i}_{j}"
            elif i == 0 and j != 0: 
                prob += u[j] <= (n - 1) * x[0, j] + (n - 2) * (1 - x[0, j]), f"MTZ_from_0_{j}" 
            elif j == 0 and i != 0: 
                prob += u[i] >= 1 - (n - 1) * (1 - x[i, 0]), f"MTZ_to_0_{i}" 
    
    # 4. Extra Constraints: il must be visited before jl
    for idx, (il, jl) in enumerate(constraints): # Sử dụng enumerate để đảm bảo tên duy nhất
        prob += u[il] <= u[jl] - 1, f"Constraint_{il}_before_{jl}_idx{idx}"

    # Objective function
    prob += pulp.lpSum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None), "Total_Cost"

    # Solve and extract solution
    prob.solve()

    tour = []
    current_city = 0
    tour.append(current_city)
    
    visited_in_extraction = {0}
    
    for _ in range(n - 1): 
        found_next = False
        for j in range(n):
            if current_city != j and x[current_city, j].varValue is not None and x[current_city, j].varValue > 0.99 and j not in visited_in_extraction:
                current_city = j
                tour.append(current_city)
                visited_in_extraction.add(current_city)
                found_next = True
                break
        if not found_next:
            break 
    
    return tour

#test

from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(6, 11)
    cost_matrix = create_cost(n)
    constraints = [(1, 3), (4, 2), (n-1, 1), (n-2, 2)]
    tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    print(f'Tour:{tour}')
    print(f'Cost of your tour: {tour_cost}')
    for i in range(n):
        num = sum([1 if j == i else 0 for j in tour])
        assert  num == 1, f'Vertex {i} repeats {num} times in tour'
    for (i, j) in constraints:
        assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (10 points)')