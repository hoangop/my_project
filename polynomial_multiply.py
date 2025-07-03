from numpy.fft import fft, ifft
from numpy import real
import numpy as np

def polynomial_multiply(a_coeff_list, b_coeff_list):
    n = len(a_coeff_list)
    m = len(b_coeff_list)
    
    # Size needed to hold the product of two polynomials
    size = n + m - 1

    # Pad a and b with zeros to reach the correct length
    a_pad = a_coeff_list + [0] * (size - n)
    b_pad = b_coeff_list + [0] * (size - m)

    # Compute FFT
    A_fft = fft(a_pad)
    B_fft = fft(b_pad)

    # Multiply corresponding coefficients in the frequency domain
    C_fft = A_fft * B_fft

    # Compute inverse FFT to return to the time domain
    c = ifft(C_fft)

    # Take the real part and cast to list of floats
    result = [real(x) for x in c]

    return result


