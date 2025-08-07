from pulp import *
from random import uniform 

# Here is a useful function to implement the LHS upper bound that we need for the encoding
# (This function is from Problem 2A and is not directly used in computeApproximateSolution for 2B)
def lhsUpperBound(c_list, bounds):
    """
    Calculates the maximum possible value of the Left Hand Side (LHS) expression
    c_1*x_1 + ... + c_n*x_n given the bounds for each x_i.

    Args:
        c_list (list): List of coefficients [c_1, ..., c_n].
        bounds (list): List of pairs [(l_1, u_1), ..., (l_n, u_n)] for each variable.

    Returns:
        float: The upper bound of the LHS expression.
    """
    n = len(c_list)
    assert len(bounds) == n
    # To maximize c_j * x_j:
    # If c_j > 0, x_j should be at its upper bound u_j.
    # If c_j < 0, x_j should be at its lower bound l_j.
    # If c_j == 0, it contributes 0.
    upper_bnd = sum([(cj * lj) if cj < 0 else (cj * uj) for (cj, (lj, uj)) in zip(c_list, bounds)])
    return upper_bnd

def solveForMaximumInequalitySatisfaction(n, m, c_matrix, d_values, bounds):
    # Always check pre-conditions: saves so much time later
    assert len(c_matrix) == m
    assert all(len(c_list) == n for c_list in c_matrix)
    assert len(d_values) == m
    assert len(bounds) == n
    assert all(lj <= uj for (lj, uj) in bounds)

    # Create the LP problem (Maximization problem)
    prob = LpProblem("Maximum Inequality Satisfaction", LpMaximize)

    # Decision Variables:
    x_vars = LpVariable.dicts("x", range(n), cat='Continuous')
    w_vars = LpVariable.dicts("w", range(m), cat='Binary')

    prob += lpSum(w_vars[j] for j in range(m)), "Number of Satisfied Inequalities"

    # Constraints:
    for i in range(n):
        l_i, u_i = bounds[i]
        prob += (x_vars[i] >= l_i), f"Lower_Bound_x_{i}"
        prob += (x_vars[i] <= u_i), f"Upper_Bound_x_{i}"

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

def computeApproximateSolution(n, m, c_matrix, d_values):
    assert n >= 1
    assert len(c_matrix) == m
    assert all(len(c_list) == n for c_list in c_matrix)
    assert len(d_values) == m

    # Step 1: Generate random numbers r_1, ..., r_n in [-1, 1]
    r_values = [uniform(-1, 1) for _ in range(n)]

    # Step 2: Transform each n-variable inequality into a 1-variable inequality: C_j' * x <= d_j
    upper_bound_ineqs = [] 
    lower_bound_ineqs = [] 
    
    always_satisfied_fixed_count = 0 
    
    for j in range(m):
        C_prime_j = sum(c_matrix[j][i] * r_values[i] for i in range(n))

        if abs(C_prime_j) < 1e-9:  
            if d_values[j] >= 0:
                always_satisfied_fixed_count += 1
        else:
            val = d_values[j] / C_prime_j
            if C_prime_j > 0:
                upper_bound_ineqs.append((val, j))
            else:
                lower_bound_ineqs.append((val, j))

    # Step 3: Find the optimal single variable 'x_val' for the 1-variable problem
    
    best_k_single_var = always_satisfied_fixed_count
    best_x_single_var = 0.0 

    candidate_x_values_for_1D = []

    # Candidate 1: Try to satisfy all 'x <= val' inequalities.
    if upper_bound_ineqs:
        x_candidate_1 = min([item[0] for item in upper_bound_ineqs])
        candidate_x_values_for_1D.append(x_candidate_1)
    
    # Candidate 2: Try to satisfy all 'x >= val' inequalities.
    if lower_bound_ineqs:
        x_candidate_2 = max([item[0] for item in lower_bound_ineqs])
        candidate_x_values_for_1D.append(x_candidate_2)

    # If there are candidates, evaluate them
    if candidate_x_values_for_1D:
        current_max_k = -1 
        current_best_x = 0.0

        for x_candidate in candidate_x_values_for_1D:
            satisfied_count_for_candidate = always_satisfied_fixed_count

            for (val, original_index) in upper_bound_ineqs:
                if x_candidate <= val + 1e-9: # Add tolerance for float comparison
                    satisfied_count_for_candidate += 1

            for (val, original_index) in lower_bound_ineqs:
                if x_candidate >= val - 1e-9: # Add tolerance
                    satisfied_count_for_candidate += 1
            
            if satisfied_count_for_candidate > current_max_k:
                current_max_k = satisfied_count_for_candidate
                current_best_x = x_candidate
        
        best_k_single_var = current_max_k
        best_x_single_var = current_best_x
    
    # Step 4: Construct the n-variable solution (x_1, ..., x_n)
    x_solution = [r_values[i] * best_x_single_var for i in range(n)]

    return best_k_single_var, x_solution


#test
def testSolution(n, m, c_matrix, d_values, x_values):
    # always check pre-conditions: saves so much time later
    assert len(c_matrix) == m
    assert all(len(c_list) == n for c_list in c_matrix)
    assert len(d_values) == m
    assert len(x_values) == n
     # Check how many inequalities satisfied
    num_ineqs = 0
    for (c_list, d) in zip(c_matrix, d_values):
        if sum([cj * xj for (cj, xj) in zip(c_list,x_values )]) <= d+1E-3:
            num_ineqs = num_ineqs + 1
    assert num_ineqs >= m/2, f' Half number of inequalities to be sat: {m/2} your solution satisfies: {num_ineqs} inequalities '
    print('Test Passed')
    return 
        
        

from random import uniform, randint, seed
## Warning: these are large instances. If your solution takes more than 120 seconds, then 
## chances are that you will not receive any credit for this problem.
def gen_random_instance(n, m):
    c_matrix = [ [randint(-5, 5) for i in range(n)] for j in range(m)]
    d_values = [randint(-10,10) for i in range(m)]
    return (c_matrix, d_values)

seed(100001)

print('Test # 1')
n = 10
m = 55
(c_matrix, d_values) = gen_random_instance(n, m)
(k, x_values) = computeApproximateSolution(n, m, c_matrix, d_values)
print(k)
print(x_values)
testSolution(n, m, c_matrix, d_values, x_values)


print('Test # 2')
n = 35
m = 230
(c_matrix, d_values) = gen_random_instance(n, m)
(k, x_values) = computeApproximateSolution(n, m, c_matrix, d_values)
print(k)
print(x_values)
testSolution(n, m, c_matrix, d_values, x_values)

print('Test # 3')
n = 100
m = 550
(c_matrix, d_values) = gen_random_instance(n, m)
(k, x_values) = computeApproximateSolution(n, m, c_matrix, d_values)
print(k)
print(x_values)
testSolution(n, m, c_matrix, d_values, x_values)

print('Test # 4')
n = 80
m = 900
(c_matrix, d_values) = gen_random_instance(n, m)
(k, x_values) = computeApproximateSolution(n, m, c_matrix, d_values)
print(k)
print(x_values)
testSolution(n, m, c_matrix, d_values, x_values)

print('Test # 5')
n = 70
m = 445
(c_matrix, d_values) = gen_random_instance(n, m)
(k, x_values) = computeApproximateSolution(n, m, c_matrix, d_values)
print(k)
print(x_values)
testSolution(n, m, c_matrix, d_values, x_values)

print('15 points!')