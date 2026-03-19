from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector

import numpy as np

from quantum.ansatz import tensor_ring


def encoding_features_h_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)

    return qc

def encoding_features_h_rx_half_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.rx(feature_params[i], i)
        qc.ry(feature_params[i]/2, i)

    return qc

def encoding_features_h_ry_rx(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)
        qc.rx(feature_params[i], i)

    return qc

def encoding_features_ry_rz_two_data(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits * 2)
    j = 0
    for i in range(num_qubits):
        qc.ry(feature_params[j], i)
        j += 1
        qc.rz(2 * feature_params[j], i)
        j += 1

    return qc

def encoding_features_ry_rz_two_data_cnot(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits * 2)
    j = 0
    for i in range(num_qubits):
        qc.ry(feature_params[j], i)
        j += 1
        qc.rz(feature_params[j], i)
        j += 1

    for i in range(num_qubits):
        qc.cx(i, (i+1) % num_qubits)

    return qc

def encoding_features_h_ry_rx_v2(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)
    j = 0
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[j], i)
        qc.rx(-feature_params[j], i)
        j+=1

    return qc

def encoding_features_h_rx_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.rx(feature_params[i], i)
        qc.ry(feature_params[i], i)

    return qc

def encoding_features_h_ry_rz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)
        qc.rz(feature_params[i], i)

    return qc

def encoding_features_h_rz_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.rz(feature_params[i], i)
        qc.ry(feature_params[i], i)

    return qc

def encoding_features_ry_rz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits*2)
    j = 0
    for i in range(num_qubits):
        qc.rx(feature_params[j], i)
        j += 1
        qc.rz(feature_params[j], i)
        j += 1

    return qc

def encoding_features_h_u(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.u(feature_params[i],feature_params[i], -feature_params[i], i)

    return qc

def encoding_features_h_rv(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.rv(feature_params[i], feature_params[i], np.pi , i)

    return qc

def encoding_features_h_rv_v1(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.rv(feature_params[i], 2*feature_params[i], np.pi , i)

    return qc

def encoding_features_h_rv_v2(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits * 2)
    j = 0
    for i in range(num_qubits):
        qc.h(i)
        qc.rv(feature_params[j] / np.sqrt(2), feature_params[j+1], 0 , i)
        j += 2

    return qc

def standard_re_uploading(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.ry(feature_params[i], i)

    tn = tensor_ring(num_qubits)

    qc.compose(tn, inplace=True)
    for i in range(num_qubits):
        qc.rx(feature_params[i], i)

    tn = tensor_ring(num_qubits, prefix='φ')
    qc.compose(tn, inplace=True)

    return qc


def partial_encoding(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    feature_params = ParameterVector('x', (num_qubits-1) * 2)
    idx = 0
    for q1, q2 in zip(qubits[:-1], qubits[1:]):
        qc.ry(feature_params[idx], q1)
        idx += 1
        qc.ry(feature_params[idx], q2)
        idx += 1

        qc.compose(
            TwoLocal(
                2,
                rotation_blocks='ry',
                entanglement='reverse_linear',
                entanglement_blocks='cx',
                parameter_prefix=f'θ_{idx/2}',
                insert_barriers=True,
                reps=1
            ),
            qubits=[q1, q2],
            inplace=True
        )

        qc.cx(q1, q2)
        qc.barrier()

    return qc