def get_continued_fraction(lst):
    """
    Tính toán liên phân số từ một danh sách các số tự nhiên.
    Đầu vào: lst = [a_0, a_1, ..., a_{n-1}]
    Trả về: một cặp số nguyên (tử số, mẫu số) của biểu thức:
             1 / (a_0 + 1 / (a_1 + 1 / (... + 1 / a_{n-1})))
    """
    assert len(lst) >= 1
    
    # Bắt đầu từ phần trong cùng của liên phân số (cuối danh sách)
    # Giá trị ban đầu là a_{n-1}, có thể viết dưới dạng phân số là lst[-1] / 1
    numerator = lst[-1]
    denominator = 1
    
    # Lặp ngược từ phần tử kế cuối về đầu danh sách
    # range(len(lst) - 2, -1, -1) sẽ đi từ index n-2 về 0
    for i in range(len(lst) - 2, -1, -1):
        # Lưu lại tử số cũ trước khi cập nhật
        old_numerator = numerator
        
        # Cập nhật tử số và mẫu số theo công thức: a_i + 1 / (tử/mẫu)
        # a_i + mẫu / tử = (a_i * tử + mẫu) / tử
        numerator = lst[i] * old_numerator + denominator
        denominator = old_numerator
        
    # Kết quả cuối cùng là nghịch đảo của phân số ta đã tính được
    # vì biểu thức tổng thể có dạng 1 / (...)
    return (denominator, numerator)


# Test cases
(n4, d4) = get_continued_fraction([5])
print(f'Test # 0: {n4}/{d4}')
assert n4 == 1 and d4 == 5

(n1, d1) = get_continued_fraction([1, 2, 2])
print(f'Test # 1: {n1}/{d1}')
assert n1 == 5 and d1 == 7

(n2, d2) = get_continued_fraction([1, 2, 1, 2, 1])
print(f'Test # 2: {n2}/{d2}')
assert n2 == 11 and d2 == 15

(n3, d3) = get_continued_fraction([1, 1, 1, 1, 1, 1])
print(f'Test # 3: {n3}/{d3}')
assert n3 == 8 and d3 == 13

print("\nTất cả các test cases đã pass!")