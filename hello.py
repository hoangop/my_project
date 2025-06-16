# hello.py

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Dùng AerSimulator cho Qiskit mới

# Tạo circuit đơn giản
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Khởi chạy simulator
sim = AerSimulator()
tqc = transpile(qc, sim)
job = sim.run(tqc, shots=1024)
res = job.result()
counts = res.get_counts()

print("Counts:", counts)
