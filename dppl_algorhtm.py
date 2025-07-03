def extend_truth_assignment(truth_assign, j, b):
    """Mở rộng một phép gán chân lý bằng cách thêm (biến j -> giá trị b)."""
    truth_assign[j] = b
    return truth_assign
    
def forget_var_in_truth_assign(truth_assign, j):
    """Loại bỏ một biến khỏi từ điển phép gán chân lý."""
    if j in truth_assign:
        del truth_assign[j]
    return truth_assign

# Giả định sự tồn tại của lớp SATInstance với cấu trúc như sau để mã có thể chạy được.
# Đây là phần giả định để minh họa. Anh chỉ cần thay thế bằng lớp SATInstance thực tế của mình.
class SATInstance:
    # Constructor: provide n the number of variables and
    # an initial list of clauses.
    # Note that variable numbers will go from 1 to n inclusive.
    # we can add clauses using the add_clause method.
    def __init__(self, n, clauses):
        self.n = n
        self.m = len(clauses)
        self.clauses = clauses
        assert self.is_valid()
    # is_valid
    # Check if all clauses are correct.
    # literals in each clause must be between 1 and n or -n and -1 
    def is_valid(self):
        assert self.n >= 1
        assert self.m >= 0
        for c in self.clauses:
            for l in c:
                assert (1 <= l and l <= self.n) or (-self.n <= l and l <= -1)
        return True
    
    # add_clause
    # Add a new clause to the list of clauses
    def add_clause(self, c):
        #check the clause we are adding.
        for l in c:
            assert (1 <= l and l <= self.n) or (-self.n <= 1 and l <= -1)
        self.clauses.append(c)
    
    ## Function: evaluate_literal
    # Evaluate a literal against a partial truth assignment
    # return 0 if the partial truth assignment does not have the variable corresponding to the literal
    # return 1 if the partial truth assignment has the variable and the literal is true
    # return -1 if the partial truth assignment has the variable and the literal is false
    def evaluate_literal(self, partial_truth_assignment, literal):
        var = abs(literal) # literal may be negated. First remove any negation using abs
        if var not in partial_truth_assignment:
            return 0
        v = partial_truth_assignment[var]
        if 1 <= literal <= self.n:
            return 1 if v else -1
        else:
            return -1 if v else 1
    
    ## TODO: Write your code here
    # Function: evaluate
    # See description above: partial_truth_assignment is a dictionary from 1 .. n to true/false.
    # since it is partial, we may have variables with no truth assignments.
    # use evaluate_literal function as a useful primitive
    # return +1 if the formula is already satisfied under partial_truth_assignment: i.e, all clauses are true
    # return 0 if formula is indeterminate under partial_truth_assignment, all clauses are true or unresolved and at least one clause is unresolved.
    # return -1 if formula is already violated under partial_truth_assignment, i.e, at least one clause is false
    def evaluate(self, partial_truth_assignment):
        # my code here
        overall_formula_has_unresolved_clauses = False

        # Loop inside every clause of instance SAT.
        for clause in self.clauses:
            clause_satisfied_by_assignment = False
            clause_has_unresolved_literal = False

            # Evaluate every literal in current clause.
            for literal in clause:
                literal_status = self.evaluate_literal(partial_truth_assignment, literal)

                if literal_status == 1:
                    clause_satisfied_by_assignment = True
                    break 
                elif literal_status == 0: 
                    clause_has_unresolved_literal = True

            if clause_satisfied_by_assignment:
                continue
            elif clause_has_unresolved_literal:
                overall_formula_has_unresolved_clauses = True
            else:
                return -1

        if overall_formula_has_unresolved_clauses:
            return 0
        else:
            return 1     


def dpll_algorithm(formula, partial_truth_assign, j):
    print("j is "+str(j))
    assert 1 <= j and j <= formula.n
    assert j not in partial_truth_assign
    # your code here
    if j > formula.n:
        eval_result = formula.evaluate(partial_truth_assign)
        if eval_result == 1:
            return (True, partial_truth_assign)
        else:
            return (False, None)

    extend_truth_assignment(partial_truth_assign, j, True)
    eval_result = formula.evaluate(partial_truth_assign)
    if eval_result == 1:
        return (True, partial_truth_assign)
    if eval_result == 0:
        (result, final_assignment) = dpll_algorithm(formula, partial_truth_assign, j + 1)
        if result:
            return (True, final_assignment)
    forget_var_in_truth_assign(partial_truth_assign, j)
    extend_truth_assignment(partial_truth_assign, j, False)
    eval_result = formula.evaluate(partial_truth_assign)
    if eval_result == 1:
        return (True, partial_truth_assign)
    if eval_result == 0:
        (result, final_assignment) = dpll_algorithm(formula, partial_truth_assign, j + 1)
        if result:
            return (True, final_assignment)
    forget_var_in_truth_assign(partial_truth_assign, j)
    return (False, None)


def solve_formula(formula):
    """
    Hàm khởi tạo để giải một công thức SAT.
    """
    return dpll_algorithm(formula, {}, 1)


#TEST

print('-- formula 1 --')
f1 = SATInstance(4, [ [ 1, 2, -4], [-2, -3, 1], [-1, -2, -3] ])
(e, t) = solve_formula(f1)
print(e, t)
assert e, 'f1 should be satisfiable'
assert t != None, 'does not return a truth assignment'
assert f1.evaluate(t) == 1, 'Truth assignment does not evaluate to expected value of true'

print('-- formula 2 -- ')
f2 = SATInstance(5, [[1,2,-5],[-4,-2,-1], [1, 3, 5], [-1, -5, -2], [1, 2, -4]])
(e2, t2) = solve_formula(f2)
print(e2, t2)
assert e2, 'f2 must be satisfiable'
assert t2 != None, 'does not return a truth assignment'
assert f2.evaluate(t2) == 1, 'Truth assignment does not evaluate to expected value of true'

print('--formula 3 --')
f3 = SATInstance(5, [[1, 2, -5, -4], [1, 2, -5, 4], [-1], [-2,-5], [5]])
(e3, t3) = solve_formula(f3)
print(e3, t3)
assert not e3, 'f3 is unsatisfiable'
assert t3 == None

print('--formula 4--')
f4 = SATInstance(10, [
  [-1, -5, -4, 8],
  [1, 5, 8, 2],
   [2, 1, 3, 9],
    [-2, 4, 5, 6, -7],
    [-1, 2, -1, 7, 8],
    [2, -3, 1, 4, 9 ],
    [1, 10],
    [-10],
    [1, 5, 8, 3, 10]
])

(e4, t4) = solve_formula(f4)
print(e4, t4)
assert e4, 'f4 must be satisfiable'
assert t4 != None, 'does not return a truth assignment'
assert f4.evaluate(t4) == 1, 'Truth assignment does not evaluate to expected value of true'

print('--formula 5--')
f5 = SATInstance(16,[
     [1, 2], [-2 , -4],[3, 4], [-4, -5], [5, -6], [6, -7], [6, 7], [7, -16],
     [8, -9],[8, -14], [9, 10], [9, -10], [-10, -11], [10, 12], [11, 12], [13, 14],
     [14, -15], [15, 16]])
(e5, t5) = solve_formula(f5)
print(e5, t5)
assert e5, 'f5 is satisfiable'
assert t5 != None
assert f5.evaluate(t5) == 1, 'Truth assignment does not evaluate to expected value of true'

print('All tests passed: 20 points')