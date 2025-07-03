from pulp import * # get all the definitions from pulp library 

def solve_candy_knapsack(n, candy_number_limits, candy_prices, candy_calories, N, Cmin, Cmax):
    assert len(candy_number_limits) == n, 'size mismatch'
    assert len(candy_prices) == n, 'size mismatch for prices'
    assert len(candy_calories) == n, 'size mismatch for list of calories'
    assert N >= 1, 'total number of candies per box must be 1 or more'
    assert Cmin <= Cmax, 'minimum calories is greater than the maximum calories'
    prob = LpProblem('Candy Knapsack', LpMinimize)
    # add decision variables
    # make a list of n decision variables, one for each candy. When declaring the variable, automatically declare
    # its lower bound to be 0 and upper bound to be ki from the candy_number_limits array
    # also declare the category (cat) of the variable to be integers.
    dVars = [LpVariable(f'x{i}',lowBound=0, upBound=ki, cat='Integer') for (i, ki) in enumerate(candy_number_limits)]
    # Now set the objective
    prob += lpSum([pj*xj for (pj,xj) in zip(candy_prices, dVars)])
    # Now add the constraints
    prob += lpSum(dVars) <= N # constraints on number of candies per box
    calories_of_candies = lpSum([cj*xj for (cj,xj) in zip(candy_calories, dVars)])
    prob += calories_of_candies <= Cmax
    prob += calories_of_candies >= Cmin
    status = prob.solve()
    if status == constants.LpStatusInfeasible:
        print('Problem is infeasible')
        return
    elif status == constants.LpStatusUnbounded:
        print('Problem is unbounded. Cannot proceed')
        return
    else:
        assert status == constants.LpStatusOptimal, 'Something went wrong while solving since status is either undefined or unsolved'
        # extract values
        print('Success: optimal answer found')
        solution_values = [x.varValue for x in dVars]
        for (j, svj) in enumerate(solution_values):
            print(f'\t Candy Type # {j}: {svj} candies')
        print(f'Total Cost: {sum([(pj*svj) for (pj, svj) in zip(candy_prices, solution_values)])}')
        print(f'Calories: {sum([cj*xj for (cj,xj) in zip(candy_calories, solution_values)])}')