from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
def quantum_teleport(qc, b0, b1, cbit): 
    # qc is a quantum circuit instance 
    # b0 and b1 are indices of two bits. 
    # cbit is a classical bit that you will need for the partial mmeasurement at step 4.
    # assume b0 != b1
    # implement the circuit to "teleport" the state of q1 to q2.
    # do not introduce any ancillary qubits.
    # no need to return anything: you will be mutating the circuit qc.
    # My code here
    
    qc.h(b1)

    # Step 2: CNOT with control b1, target b0
    qc.cx(b1, b0)

    # Step 3: CNOT with control b0, target b1
    qc.cx(b0, b1)

    # Step 4: Measure b0 into cbit
    qc.measure(b0, cbit)

# Create quantum and classical registers
#b = QuantumRegister(2, 'b')
#c = ClassicalRegister(1,'z')
#qc = QuantumCircuit(b, c)

# Apply teleportation
#quantum_teleport(qc, b[0], b[1], c[0])

# Visualize circuit
#qc.draw('mpl', style='iqp')

#-----------------------------------------------------

# Test 1. We will create a single qubit circuit and copy it over.
# 
from numpy import pi
from qiskit import Aer, transpile
from IPython.display import display


b = QuantumRegister(2, 'b')
c = ClassicalRegister(2,'z')
qc_test1 = QuantumCircuit(b, c)

qc_test1.r(2*pi/8, 0.0, b[0]) # apply a pi/8 rotation
# b[0] should be prepared in state 
# cos(pi/8) |0> - sin(pi/8) |1>
qc_test1.barrier()
# Teleport b[0] to b[1]
quantum_teleport(qc_test1, b[0], b[1], c[1]) # teleport 
qc_test1.barrier()
qc_test1.measure(b[1], c[0]) # measure
display(qc_test1.draw('mpl', style='iqp'))
# test 
from qiskit.visualization import plot_histogram 
simulator = Aer.get_backend('aer_simulator')
circ = transpile(qc_test1, simulator) 

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print(f'Result counts from 1024 simulations: {counts}')
# collect the counts
# roughly we should get measurement of b[0] = 0, b[0] = 1 with 50% probability
# probability of b[1] = 0 and b[1] = 1 must be in the ratio tan^2(pi/8) ~ 0.17157

b0_0_count = counts['00'] + counts['01']
b0_1_count = counts['10'] + counts['11']
print(f' b0 = |0> count is {b0_0_count} and b0=|1> count is {b0_1_count} : must be roughly equal')
assert b0_0_count >= 0.85 * 512 and b0_0_count <= 1.15 * 512

b1_0_count = counts['00'] + counts['10']
b1_1_count = counts['01'] + counts['11']

print(f' b1 = |0> count is {b1_0_count} and b1=|1> count is {b1_1_count} :expected  b1 = |1> happens roughly 17% of the time, and b1=|0> 83% ')
assert b1_0_count >= 0.7 * 1024 and b1_0_count <= 0.95 * 1024

print('Test passed')
