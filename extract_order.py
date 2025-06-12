# Helper function from Part A
def get_continued_fraction(lst):
    assert len(lst) >= 1
    # your code here
  # Start with the innermost term a_{n-1} as a fraction.
    numerator = lst[-1]
    denominator = 1
    
    # Iterate backwards from the second-to-last element.
    for i in range(len(lst) - 2, -1, -1):
        # Store the old numerator.
        old_numerator = numerator
        
        # Update the fraction using the formula: a_i + 1 / (num/den)
        # which simplifies to: (a_i * num + den) / num
        numerator = lst[i] * old_numerator + denominator
        denominator = old_numerator
        
    # The final expression is the reciprocal of the calculated fraction.
    return (denominator, numerator)

# Helper function from Part B
def make_continued_fraction(a, b):
    assert a > 0
    assert a <= b
    # your code here
    
    coeffs = []
    
    # This loop is a variant of the Euclidean algorithm
    while a != 0:
        # The next coefficient is the integer part of b/a
        quotient = b // a
        coeffs.append(quotient)
        
        # Update for the next iteration: the new fraction is (b%a)/a
        # which means new_a = b % a and new_b = a
        a, b = b % a, a
        
    return coeffs

# Provided modular exponentiation function
def modular_exponentiate(a, k, n): # Computes a^k mod n
    """Efficiently computes (a^k) % n."""
    mu = a
    res = 1
    while k > 0:
        if k % 2 == 1:
            res = (res * mu) % n   
        mu = (mu * mu) % n
        k = k // 2
    return res

# Main function for Part C
def extract_order(meas, m, a, n):
   
    if meas == 0:
        return None # Cannot proceed if measurement is 0
    
    numerator = meas
    denominator = 2**m   
   
    coeffs = make_continued_fraction(numerator, denominator)
    
    for i in range(1, len(coeffs) + 1):
        sub_coeffs = coeffs[0:i]
        j_prime, r_prime = get_continued_fraction(sub_coeffs)
        
        # Heuristic: The order r must be less than n.
        if r_prime < n:
            # Test r_prime
            if modular_exponentiate(a, r_prime, n) == 1:
                return r_prime
            
            # Test 2 * r_prime
            if 2 * r_prime < n and modular_exponentiate(a, 2 * r_prime, n) == 1:
                return 2 * r_prime
            
            # Test 3 * r_prime
            if 3 * r_prime < n and modular_exponentiate(a, 3 * r_prime, n) == 1:
                return 3 * r_prime

    return None

# Test cases
# Test cases from the problem description
r1 = extract_order(75, 7, 5, 91)
print(f"Test 1 (meas=75): Found order = {r1}")
assert r1 == 12

r2 = extract_order(53, 7, 5, 91)
print(f"Test 2 (meas=53): Found order = {r2}")
assert r2 == 12

r3 = extract_order(96, 7, 5, 91)
print(f"Test 3 (meas=96): Found order = {r3}")
assert r3 == 12

r4 = extract_order(32, 7, 5, 91)
print(f"Test 4 (meas=32): Found order = {r4}")
assert r4 == 12

r5 = extract_order(64, 7, 5, 91)
print(f"Test 5 (meas=64): Found order = {r5}")
assert r5 is None

r6 = extract_order(11, 7, 5, 91)
print(f"Test 6 (meas=11): Found order = {r6}")
assert r6 == 12

print("\nAll test cases passed! ✅")