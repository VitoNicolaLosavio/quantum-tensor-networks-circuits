from qiskit import QuantumCircuit
from qiskit.circuit.library import  TwoLocal
from qiskit.circuit import ParameterVector, Parameter


def MPS(num_qubits, rotation_blocks='ry', reps=1):
    """
    Constructs a Matrix Product State (MPS) quantum circuit.
    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    # Iterate over adjacent qubit pairs
    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            TwoLocal(
                2,
                rotation_blocks=rotation_blocks,
                entanglement='reverse_linear',
                entanglement_blocks='cx',
                parameter_prefix=f'θ_{i},{j}',
                insert_barriers=True,
                reps=reps
            ), [i, j],
                   inplace=True)
        qc.barrier(
        )

    return qc

def _generate_tree_tuples(n):
    """
    Generate a list of tuples representing the tree structure
    of consecutive numbers up to n.
    """
    tuples_list = []
    indices = []

    # Generate initial tuples with consecutive numbers up to n
    for i in range(0, n, 2):
        tuples_list.append((i, i + 1))

    indices += [tuples_list]

    # Perform iterations until we reach a single tuple
    while len(tuples_list) > 1:
        new_tuples = []

        # Generate new tuples by combining adjacent larger numbers
        for i in range(0, len(tuples_list), 2):
            new_tuples.append((tuples_list[i][1], tuples_list[i + 1][1]))

        tuples_list = new_tuples
        indices += [tuples_list]

    return indices


def TTN(num_qubits, cut_last_layer=False, reps=1, rotation_blocks='ry'):
    """
    Constructs a Tree Tensor Network (TTN) quantum circuit.
    """
    qc = QuantumCircuit(num_qubits)

    # Compute qubit indices
    assert num_qubits & (
            num_qubits -
            1) == 0 and num_qubits != 0, "Number of qubits must be a power of 2"

    indices = _generate_tree_tuples(num_qubits)
    if cut_last_layer:
        indices = indices[:-1]

    # Iterate over each layer of TTN indices
    for layer_indices in indices:
        for i, j in layer_indices:
            qc.compose(
                TwoLocal(
                    2,
                    rotation_blocks=rotation_blocks,
                    entanglement='reverse_linear',
                    entanglement_blocks='cx',
                    parameter_prefix=f'λ_{i},{j}',
                    insert_barriers=True,
                    reps=reps
                ), [i, j],
                inplace=True)

        qc.barrier()

    return qc

def MERA(num_qubits, reps=1, rotation_blocks='ry'):
    assert num_qubits == 6, "Number of qubits must be 6 for use this implementation"

    qc = QuantumCircuit(num_qubits)

    # split organized by layers
    layers_couples = [[(1,2), (3,4)], [(0,1), (2, 3), (5, 4)], [(4, 1)], [(2,1), (3, 4)], [(1, 4)]]

    for idx, layer in enumerate(layers_couples):
        for i, j in layer:
            qc.compose(
                TwoLocal(
                    2,
                    rotation_blocks=rotation_blocks,
                    entanglement='reverse_linear',
                    entanglement_blocks='cx',
                    parameter_prefix=f'λ_layer_{idx}_{i},{j}',
                    insert_barriers=True,
                    reps=reps
                ),
                [i, j],
                inplace=True
            )
        qc.barrier()

    return qc


def CMPS(n):
    qubits = list(range(n))
    ansatz = QuantumCircuit(n)

    idx = 0

    qubit_coupled = list(zip(qubits[::1], qubits[1::]))
    for q_1, q_2 in qubit_coupled:
        ansatz.rx(Parameter(f'θ[{idx}]'), q_1)
        idx += 1
        ansatz.ry(Parameter(f'θ[{idx}]'), q_2)
        idx += 1
        ansatz.cx(q_1, q_2)

    return ansatz

def CTTN(n):
    ansatz = QuantumCircuit(n)

    idx = 0
    layer = _generate_tree_tuples(n)

    for qubits_coupled in layer:
        for q_1, q_2 in qubits_coupled:
            ansatz.rx(Parameter(f'φ[{idx}]'), q_1)
            idx += 1
            ansatz.rx(Parameter(f'φ[{idx}]'), q_2)
            idx += 1
            ansatz.cx(q_1, q_2)
    return ansatz

def tensor_ring(num_qubits, rotation_blocks='ry', prefix='θ', reps=1):
    """
    Constructs a Matrix Product State (MPS) with periodic boundary circuit, forming a ring-like structure

    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    # Iterate over adjacent qubit pairs
    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            TwoLocal(
                    2,
                    rotation_blocks=rotation_blocks,
                    entanglement='reverse_linear',
                    entanglement_blocks='cx',
                    parameter_prefix=f'{prefix}_{i},{j}',
                    insert_barriers=True,
                    reps=reps
                ), [i, j],
                   inplace=True)
        qc.barrier(
        )

    qc.compose(TwoLocal(
                    2,
                    rotation_blocks=rotation_blocks,
                    entanglement='reverse_linear',
                    entanglement_blocks='cx',
                    parameter_prefix=f'{prefix}_{num_qubits-1},{0}',
                    insert_barriers=True,
                    reps=reps
                ), [num_qubits-1, 0],
               inplace=True)
    qc.barrier(
    )

    return qc


def construct_tensor_ring_ansatz_circuit(num_qubits, cut_last_layer=False)-> (QuantumCircuit, list):
    # Function for the construction of the TR+TTN ansatz
    ansatz = QuantumCircuit(num_qubits)

    last_ttn_qubit = _generate_tree_tuples(num_qubits)
    if cut_last_layer:
        last_ttn_qubit = last_ttn_qubit[:-1][-1]
    else:
        last_ttn_qubit = last_ttn_qubit[-1]

    ttn = TTN(num_qubits, cut_last_layer=cut_last_layer, reps=1).decompose()
    tr = tensor_ring(num_qubits, prefix='λ_TR', reps=1).decompose()

    ansatz.compose(tr, range(num_qubits), inplace=True)
    ansatz.compose(ttn, range(num_qubits), inplace=True)
    

    return ansatz, last_ttn_qubit

def interaction_layer_circuit(n_qubit=4, output_shape=2):
    tot_qubits = n_qubit + output_shape

    qc = QuantumCircuit(tot_qubits)

    ansatz = QuantumCircuit(tot_qubits, output_shape)
    tn = construct_tensor_ring_ansatz_circuit(n_qubit)

    ansatz.compose(tn, range(n_qubit), inplace=True)
    ansatz.barrier()

    ancilla_qubits = list(range(n_qubit, n_qubit + output_shape))
    latent_qubit = [3,1]
    for i, j in enumerate(latent_qubit):
        ansatz.cx(j, i+n_qubit)

    ansatz.barrier()

    params = ParameterVector('φ', 3 * output_shape)
    for j, a in enumerate(ancilla_qubits):
        base = 3 * j
        ansatz.rz(params[base], a)
        ansatz.ry(params[base + 1], a)
        ansatz.rx(params[base + 2], a)

    qc.compose(ansatz, range(tot_qubits), inplace=True)
    for i, anc in enumerate(ancilla_qubits):
        qc.measure(anc, i)


    return qc


def interaction_layer_base(num_qubits=8, n_ancilla=2):
    total_qubits = num_qubits + n_ancilla

    qc = QuantumCircuit(total_qubits, n_ancilla)
    tn, _ = construct_tensor_ring_ansatz_circuit(num_qubits, cut_last_layer=True)

    qc.compose(tn, range(num_qubits), inplace=True)
    qc.barrier()

    qubits_cx = [7, 5, 3, 1]

    for i in range(n_ancilla):
        qc.cx(qubits_cx[i], num_qubits + i)

    qc.barrier()

    params = ParameterVector(name='φ' ,length=n_ancilla * 3)

    for i in range(n_ancilla):
        idx = num_qubits + i
        qc.rx(params[3 * i], idx)
        qc.ry(params[3 * i + 1], idx)
        qc.rz(params[3 * i + 2], idx)

    return qc
