
import pulp

# 1. Khởi tạo bài toán:
# Tạo một bài toán tối đa hóa
prob = pulp.LpProblem("Integer Linear Programming Problem", pulp.LpMaximize)

# 2. Định nghĩa các biến:
# x1, x2, x3, x4, x5 là các biến số nguyên
# Các biến này có giới hạn dưới là -15 và giới hạn trên là 15
x1 = pulp.LpVariable("x1", -15, 15, pulp.LpInteger)
x2 = pulp.LpVariable("x2", -15, 15, pulp.LpInteger)
x3 = pulp.LpVariable("x3", -15, 15, pulp.LpInteger)
x4 = pulp.LpVariable("x4", -15, 15, pulp.LpInteger)
x5 = pulp.LpVariable("x5", -15, 15, pulp.LpInteger)

# 3. Định nghĩa hàm mục tiêu:
# max 2x1 - 3x2 + x3
prob += 2 * x1 - 3 * x2 + x3, "Objective Function"

# 4. Định nghĩa các ràng buộc:
# s.t.
# x1 - x2 + x3 <= 5
prob += x1 - x2 + x3 <= 5, "Constraint 1"
# x1 - x2 + 4x3 <= 7
prob += x1 - x2 + 4 * x3 <= 7, "Constraint 2"
# x1 + 2x2 - x3 + x4 <= 14
prob += x1 + 2 * x2 - x3 + x4 <= 14, "Constraint 3"
# x3 - x4 + x5 <= 7
prob += x3 - x4 + x5 <= 7, "Constraint 4"

# 5. Giải bài toán:
# Sử dụng trình giải mặc định của PuLP (thường là CBC)
prob.solve()

# 6. In trạng thái giải pháp:
print(f"Trạng thái: {pulp.LpStatus[prob.status]}")

# 7. In các giá trị biến tối ưu và giá trị hàm mục tiêu:
print("Giá trị biến tối ưu:")
for v in prob.variables():
    print(f"{v.name} = {v.varValue}")

optimal_value_ilp = pulp.value(prob.objective)
print(f"Giá trị tối ưu của hàm mục tiêu (ILP): {optimal_value_ilp:.2f}")

print("\n--- Giải pháp thư giãn LP ---")

# 1. Khởi tạo bài toán thư giãn LP:
# Tạo một bài toán tối đa hóa
prob_lp = pulp.LpProblem("LP Relaxation Problem", pulp.LpMaximize)

# 2. Định nghĩa các biến cho thư giãn LP:
# Các biến này là liên tục (mặc định trong PuLP)
x1_lp = pulp.LpVariable("x1_lp", -15, 15, pulp.LpContinuous)
x2_lp = pulp.LpVariable("x2_lp", -15, 15, pulp.LpContinuous)
x3_lp = pulp.LpVariable("x3_lp", -15, 15, pulp.LpContinuous)
x4_lp = pulp.LpVariable("x4_lp", -15, 15, pulp.LpContinuous)
x5_lp = pulp.LpVariable("x5_lp", -15, 15, pulp.LpContinuous)

# 3. Định nghĩa hàm mục tiêu cho thư giãn LP:
prob_lp += 2 * x1_lp - 3 * x2_lp + x3_lp, "Objective Function LP"

# 4. Định nghĩa các ràng buộc cho thư giãn LP (giống như ILP):
prob_lp += x1_lp - x2_lp + x3_lp <= 5, "Constraint 1 LP"
prob_lp += x1_lp - x2_lp + 4 * x3_lp <= 7, "Constraint 2 LP"
prob_lp += x1_lp + 2 * x2_lp - x3_lp + x4_lp <= 14, "Constraint 3 LP"
prob_lp += x3_lp - x4_lp + x5_lp <= 7, "Constraint 4 LP"

# 5. Giải bài toán thư giãn LP:
prob_lp.solve()

# 6. In trạng thái giải pháp thư giãn LP:
print(f"Trạng thái (LP Relaxation): {pulp.LpStatus[prob_lp.status]}")

# 7. In các giá trị biến tối ưu và giá trị hàm mục tiêu cho thư giãn LP:
print("Giá trị biến tối ưu (LP Relaxation):")
for v in prob_lp.variables():
    print(f"{v.name} = {v.varValue}")

optimal_value_lp = pulp.value(prob_lp.objective)
print(f"Giá trị tối ưu của hàm mục tiêu (LP Relaxation): {optimal_value_lp:.2f}")
