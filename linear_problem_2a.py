from pulp import *

# Here is a useful function to implement the LHS upper bound that we need for the encoding
def lhsUpperBound(c_list, bounds):

    n = len(c_list)
    assert len(bounds) == n
    # To maximize c_j * x_j:
    # If c_j > 0, x_j should be at its upper bound u_j.
    # If c_j < 0, x_j should be at its lower bound l_j.
    # If c_j == 0, it contributes 0.
    upper_bnd = sum([(cj * lj) if cj < 0 else (cj * uj) for (cj, (lj, uj)) in zip(c_list, bounds)])
    return upper_bnd

def solveForMaximumInequalitySatisfaction(n, m, c_matrix, d_values, bounds):

    assert len(c_matrix) == m
    assert all(len(c_list) == n for c_list in c_matrix)
    assert len(d_values) == m
    assert len(bounds) == n
    assert all(lj <= uj for (lj, uj) in bounds)

    # Create the LP problem
    prob = LpProblem("Maximum Inequality Satisfaction", LpMaximize)

    # Decision Variables:
    x_vars = LpVariable.dicts("x", range(n), cat='Continuous')

  
    w_vars = LpVariable.dicts("w", range(m), cat='Binary')

    # Objective Function: 
    prob += lpSum(w_vars[j] for j in range(m)), "Number of Satisfied Inequalities"

    # Constraints:
    for i in range(n):
        l_i, u_i = bounds[i]
        prob += (x_vars[i] >= l_i), f"Lower_Bound_x_{i}"
        prob += (x_vars[i] <= u_i), f"Upper_Bound_x_{i}"

    # Transformed Inequality Constraints:
    for j in range(m):
        lhs_expression = lpSum(c_matrix[j][i] * x_vars[i] for i in range(n))
        M_j = lhsUpperBound(c_matrix[j], bounds)

        # Add the transformed constraint
        prob += (lhs_expression <= d_values[j] * w_vars[j] + M_j * (1 - w_vars[j])), f"Inequality_{j}_Satisfaction"

    # Solve the problem
    prob.solve()

    # Prepare the result
    k = 0
    x_solution = [0.0] * n # Initialize with zeros

    # Check if an optimal solution was found using the integer value for Optimal status
    if prob.status == 1: 
        k = int(value(prob.objective)) 
        for i in range(n):
            x_solution[i] = value(x_vars[i]) 
    else:
        print(f"Warning: Problem status is not Optimal. Status: {LpStatus[prob.status]}")

    return k, x_solution


#test
def testSolution(n, m, c_matrix, d_values, bounds, x_values, k_expected):
    # always check pre-conditions: saves so much time later
    assert len(c_matrix) == m
    assert all(len(c_list) == n for c_list in c_matrix)
    assert len(d_values) == m
    assert len(bounds) == n
    assert all (lj <= uj for (lj, uj) in bounds)
    assert len(x_values) == n
    assert 0 <= k_expected <= m
    # check solution within bounds
    for i in range(n):
        (lb, ub) = bounds[i]
        assert lb <= x_values[i] <= ub, f'x_{i} fails to be within its bounds {[lb, ub]}'
    # Check how many inequalities satisfied
    num_ineqs = 0
    
    for (c_list, d) in zip(c_matrix, d_values):
        if sum([cj * xj for (cj, xj) in zip(c_list,x_values )]) <= d+1E-3:
            num_ineqs = num_ineqs + 1
    assert num_ineqs == k_expected, f' Expected number of inequalities to be sat: {k_expected} your solution satisfies: {num_ineqs} inequalities '
    print('Test Passed')
    return 
        
        
        
        

n = 5
m = 24
c_matrix = [
    [1, -1, 0, 1, -1],
    [1, 2, 0, 0, 2],
    [-1, 0, 1, 1, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, -1, -1],
    [1, 0, 0, 1, 1],
    [1, 0, -1, 1, 0],
    [0, 2, 1, 0, 2],
    [-1, 1, 1, -1, 0],
    [1, 1, 1, 0, 1],
    [-1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [-1, 1, 0, 1, -1],
    [1, -2, 0, 0, -2],
    [1, 0, 1, -1, -1],
    [1, 0, 1, 0, 1],
    [-1, 0, 0, 1, 1],
    [-1, 0, 0, 1, 1],
    [1, -1, 1, 1, 1],
    [0, -2, -1, 0, 2],
    [-1, -1, -1, -1, 0],
    [-1, 1, -1, 0, 1],
    [1, 0, 0, 1, 0],
    [-1, 0, -1, 0, -1],
]

d_list = [
    -5, 3, -4, -2, -3, -1,
    -5, 3, -4, -2, -3, -1,
     5, -3, 4, 2, 3, 1,
    5, -3, 4, 2, 3, 1,
    
]

bounds = [(-10, 10), (-10, 10), (-12, 12), (-1, 3), (3, 6)]

(k, x_values) = solveForMaximumInequalitySatisfaction(n, m, c_matrix, d_list, bounds)
testSolution(n, m, c_matrix, d_list, bounds, x_values, 18)
print('8 points')