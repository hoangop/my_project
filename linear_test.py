from pulp import *

problem = LpProblem('name', LpMinimize) # Use LpMaximize if you are maximizing an objective function
x_1 = LpVariable('x_1')
x_2 = LpVariable('x_2', 0)
x_3 = LpVariable('x_3', 0)
x_4 = LpVariable('x_4', 0)
x_5 = LpVariable('x_5', -1, 1)
problem += (x_1 + x_2 - x_3 + 2*x_4 - 3 * x_5)
problem += (x_1 - 2 * x_2 + x_3 <= 5)
problem += (2* x_2 - x_4 + x_5  <= 7)
problem += (x_1 - x_5 + 2 * x_4 <= 8)
problem.solve() # solve the problem



if problem.status == constants.LpStatusOptimal:
    print('Optimal Solution Found!!')
    # Extract the values of the decision variables.
    v_1 = x_1.varValue
    v_2 = x_2.varValue
    v_3 = x_3.varValue
    v_4 = x_4.varValue
    v_5 = x_5.varValue
    print('x_1 = {v_1}, x_2 = {v_2}, x_3 = {v_3}, x_4 = {v_4}, x_5={v_5}')
elif problem.status == constants.LpStatusUnbounded:
    print('Unbounded solution -- need more constraints')
elif problem.status == constants.LpStatusInfeasible:
    print('Problem has no feasible solution')
else: 
    print('Problem has an undefined status -- something went wrong.')