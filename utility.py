from qiskit.quantum_info import Statevector, state_fidelity
from scipy.stats import entropy
import numpy as np


def compute_expressivity_kl(circuit, num_samples=300, num_bins=75, seed=42):
    rng = np.random.default_rng(seed)
    params = list(circuit.parameters)
    if not params:
        return None

    states = []
    for _ in range(num_samples):
        values = rng.uniform(0, 2*np.pi, len(params))
        bound = circuit.assign_parameters(dict(zip(params, values)))
        states.append(Statevector.from_instruction(bound))

    fidelities = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fidelities.append(state_fidelity(states[i], states[j]))
    fidelities = np.asarray(fidelities)

    n = circuit.num_qubits
    N = 2 ** n

    hist_p, edges = np.histogram(fidelities, bins=num_bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    haar_pdf = (N - 1) * (1 - centers) ** (N - 2)
    haar_pdf /= np.sum(haar_pdf)
    hist_p /= np.sum(hist_p)

    kl = entropy(hist_p + 1e-12, haar_pdf + 1e-12)
    return {"expressivity_kl": kl, "mean_fidelity": fidelities.mean()}