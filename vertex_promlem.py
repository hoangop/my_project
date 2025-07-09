import pulp

# 1. Định nghĩa đồ thị
# Các đỉnh được đánh số từ 1 đến 7
nodes = [1, 2, 3, 4, 5, 6, 7]

# Các cạnh của đồ thị
edges = [
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 7),
    (5, 7), (6, 5), (6, 7)
]

# 2. Khởi tạo bài toán LP Relaxation
# Đây là bài toán tối thiểu hóa
prob = pulp.LpProblem("Vertex Cover LP Relaxation", pulp.LpMinimize)

# 3. Định nghĩa các biến quyết định
# Với mỗi đỉnh, tạo một biến liên tục xi nằm trong khoảng [0, 1]
x = pulp.LpVariable.dicts("x", nodes, 0, 1, pulp.LpContinuous)

# 4. Định nghĩa hàm mục tiêu
# Tối thiểu hóa tổng các giá trị xi
prob += pulp.lpSum(x[i] for i in nodes), "Total number of vertices in cover"

# 5. Định nghĩa các ràng buộc
# Với mỗi cạnh (u, v), đảm bảo rằng x[u] + x[v] >= 1
for u, v in edges:
    prob += x[u] + x[v] >= 1, f"Edge_({u},{v})_covered"

# 6. Giải bài toán
# Sử dụng trình giải mặc định của PuLP (thường là CBC)
# msg=0 để tắt thông báo chi tiết của trình giải
solver = pulp.PULP_CBC_CMD(msg=0)
prob.solve(solver)

# 7. In trạng thái giải pháp
print(f"Trạng thái giải pháp: {pulp.LpStatus[prob.status]}")

# 8. In các giá trị biến tối ưu và giá trị hàm mục tiêu
if pulp.LpStatus[prob.status] == 'Optimal':
    # code xử lý khi tối ưu:
    print("\nGiá trị biến tối ưu:")
    for node in nodes:
        print(f"x[{node}] = {x[node].varValue:.4f}") # Làm tròn để dễ đọc hơn

    optimal_objective_value = pulp.value(prob.objective)
    print(f"\nGiá trị tối ưu của hàm mục tiêu (LP Relaxation): {optimal_objective_value:.2f}")
else:
    print("Không tìm thấy giải pháp tối ưu cho LP Relaxation.")

# Kiểm tra các ràng buộc (tùy chọn, để xác minh)
print("\nKiểm tra ràng buộc:")
for i, (u, v) in enumerate(edges):
    constraint_value = x[u].varValue + x[v].varValue
    print(f"Cạnh ({u},{v}): x[{u}] ({x[u].varValue:.2f}) + x[{v}] ({x[v].varValue:.2f}) = {constraint_value:.2f} >= 1 (Hợp lệ: {constraint_value >= 1})")

