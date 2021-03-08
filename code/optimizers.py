import scipy as sp
import numpy as np

def _circuit_energy(param, circuit, hamiltonian):
	circuit.set_parameters(param)
	state = circuit()
	return sp.sparse.csr_matrix.dot(sp.sparse.csr_matrix.conjugate(state), hamiltonian(state))


class OptimizerGradientFree(Object):
	def __init__(self, hamiltonian, circuit, algorithm, param_dict):
		self.hamiltonian = hamiltonian
		self.circuit = circuit
		self.algorithm = algorithm
		self.alg_param_dict = param_dict

		return

	def optimize(self):
		res = self.algorithm(_circuit_energy, args=(circuit, hamiltonian), **self.alg_param_dict)

		return res.x