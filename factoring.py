def gcd(m, n):
    if m < n:
        (m,n) = (n,m)
    assert m > 0
    assert n >= 0
    while n != 0:
        (m, n) = (n, m%n)
    return m

def modular_exponentiate(a, k, n): # a^k \mod n
    r = a
    m = 1
    while k != 0:
        if k%2 == 1:
            m = (m * r)%n
        k = k // 2
        r = (r * r) % n
    return m

def find_order(a, n):
    r = 1
    x = a
    while x != 1:
        x = (x * a) % n
        r = r + 1
    return r

def factor_using_order_finding(a, n):
    r = find_order(a, n)
    print(f'Order of {a} w.r.t {n} = {r}')
    if r %2 == 0:
        m = modular_exponentiate(a, r//2, n)
        return gcd(m-1, n)
    print('Bad luck: odd order')
    return 1


n = 758717
a = 1998
f = factor_using_order_finding(a, n)
print(f'Found factor= {f}')
