from qiskit.circuit.library import MCPhaseGate
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi
def mark_pure_states(qc, b0 , b1, b2, b3):
    # mark the components corresponding to the pure states |0011> , |1100>, |0101>
    # your code here
    qubits = [b0, b1, b2, b3]

    # Mark |0011⟩
    qc.x(b0)
    qc.x(b1)
    qc.append(MCPhaseGate(pi, 3), [b0, b1, b2, b3])
    qc.x(b0)
    qc.x(b1)

    # Mark |1100⟩
    qc.x(b2)
    qc.x(b3)
    qc.append(MCPhaseGate(pi, 3), [b0, b1, b2, b3])
    qc.x(b2)
    qc.x(b3)

    # Mark |0101⟩
    qc.x(b0)
    qc.x(b2)
    qc.append(MCPhaseGate(pi, 3), [b0, b1, b2, b3])
    qc.x(b0)
    qc.x(b2)
    
    
b = QuantumRegister(4, 'b')
qc = QuantumCircuit(b)
mark_pure_states(qc, b[0], b[1], b[2], b[3])
qc.draw('mpl', style='iqp')