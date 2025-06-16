from qiskit import QuantumCircuit

def create_ghz_circuit(qc):
    # please implement your circuit here.
    # useful tutorials
    #  https://docs.quantum.ibm.com/build/circuit-construction
    # your code here
    # Step 1: H to qubit 0 to create superposition
    qc.h(0)
    # Step 2: CNOT to contrain qubit 1 và 2 following qubit 0 state
    qc.cx(0, 1)
    qc.cx(0, 2)
    
qc = QuantumCircuit(3) # create a quantum circuit with three qubits
create_ghz_circuit(qc) # call the code created by the student to populate the gates in the circuit
qc.measure_all() # measure the outputs
qc.draw('mpl', style="iqp") # draw the circuit