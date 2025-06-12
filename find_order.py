def gcd(m, n):
    (m, n) = max(m, n), min(m, n)
    assert m > 0
    assert n >= 0
    while n > 0:
        (m, n) = (n, m % n)
    return m

def modular_exponentiate(a, k, n):
    m = a
    res = 1
    while k > 0:
        if k % 2 == 1:
            res = (res * m) % n
        m = (m * m) % n
        k //= 2
    return res

def find_order(a, n):
    if gcd(a, n) != 1:
        return None
    r = 1
    power = a % n
    while power != 1:
        power = (power * a) % n
        r += 1
    return r

def count_hits(n):
    hits = 0
    for a in range(2, n):
        if gcd(a, n) != 1:
            continue  # skip non-coprime
        r = find_order(a, n)
        if r % 2 != 0:
            continue  # only interested in even order
        val = modular_exponentiate(a, r // 2, n)
        if (val + 1) % n != 0:
            hits += 1
    return hits


def find_possible_inputs(a, n, k, m):
    results = []
    upper_bound = 2**m
    for x in range(1, upper_bound):
        if modular_exponentiate(a, x, n) == k:
            results.append(x)
    return results    

#print(find_possible_inputs(5, 21, 5**2 % 21, 5))  # Expect [2, 8, 14, 20] when k = 4
# Hoặc đúng theo đề:
print(find_possible_inputs(5, 21, 4, 5))  # Output: [2, 8, 14, 20]
