import scipy as sp
import numpy as np

def _circuit_energy(param, circuit, hamiltonian, config):
    circuit.set_parameters(param)
    state = circuit()
    print(np.dot(np.conj(state), hamiltonian(state)).real)
    return np.dot(np.conj(state), hamiltonian(state)).real


class OptimizerGradientFree(object):
    def __init__(self, hamiltonian, circuit, algorithm, config, param_dict):
        self.hamiltonian = hamiltonian
        self.circuit = circuit
        self.algorithm = algorithm
        self.alg_param_dict = param_dict
        self.config = config

        self.alg_param_dict['bounds'] = [(-np.pi, np.pi)] * len(circuit.get_parameters())
        return

    def optimize(self):
        res = self.algorithm(_circuit_energy, args=(self.circuit, self.hamiltonian, self.config), **self.alg_param_dict)

        return res.x