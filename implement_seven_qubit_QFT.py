from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, transpile
from numpy import pi, sqrt
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt

def implement_seven_qubit_QFT(qc, b):
    """
    Hiện thực hóa mạch QFT cho 7 qubit.
    Giả định b là một thanh ghi lượng tử 7 qubit.
    """
    assert len(b) == 7
    n = 7
    
    # Áp dụng các phép quay và cổng Hadamard
    # Vòng lặp từ qubit có trọng số cao nhất (MSB) đến qubit có trọng số thấp nhất (LSB)
    for i in range(n - 1, -1, -1):
        qc.h(b[i])
        for j in range(i - 1, -1, -1):
            # Áp dụng phép quay pha có điều kiện
            qc.cp(pi / (2**(i - j)), b[j], b[i])
            
    # Thêm các cổng SWAP để đảo ngược thứ tự các qubit
    for i in range(n // 2):
        qc.swap(b[i], b[n - 1 - i])



# --- Phần mã nguồn của bạn để kiểm tra ---

# Khởi tạo các thanh ghi
b = QuantumRegister(7, 'b')
m_out = ClassicalRegister(7, 'm')
qc = QuantumCircuit(b, m_out)

# Chuẩn bị trạng thái ban đầu |ψ⟩
lst = [4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
c = 1.0/sqrt(len(lst))
state_vector = [c if i in lst else 0.0 for i in range(128)]
print('Trạng thái chồng chập ban đầu là: ', state_vector)

# Khởi tạo mạch với trạng thái |ψ⟩
qc.initialize(state_vector, b)

# Áp dụng mạch QFT 7 qubit bạn vừa hiện thực
implement_seven_qubit_QFT(qc, b)

# Đo lường các qubit
qc.measure(b, m_out)

# Vẽ mạch lượng tử
print("\nSơ đồ mạch lượng tử:")
display(qc.draw('mpl', style='iqp'))        