import scipy as sp
import numpy as np
import utils

def _circuit_energy(param, circuit, hamiltonian, config):
    circuit.set_parameters(param)
    state = circuit()
    assert np.isclose(state.conj().dot(state), 1.0)
    #for i in range(len(state)):
    #    print(utils.index_to_spin(np.array([i]), number_spins = 16), state[i])
    print(np.dot(np.conj(state), hamiltonian(state)).real)
    #print(param)
    return np.dot(np.conj(state), hamiltonian(state)).real


def gradiend_descend(energy_val, init_values, args, circuit = None, \
                     hamiltonian = None, config = None, n_iter = 100, lr = 0.01):
    for n_iter in range(n_iter):
        cur_params = circuit.get_parameters()
        state = circuit()

        grads = np.array([2 * np.dot(state.conj(), hamiltonian(circuit.derivative(i))).real \
                          for i in range(len(cur_params))])
        new_params = cur_params - lr * grads

        circuit.set_parameters(new_params)

        print('iteration: {:d}, energy = {:.7f}'.format(n_iter, _circuit_energy(new_params, *args)))
        #print(new_params)
    return circuit

def get_all_derivatives(cur_params, circuit, hamiltonian, config):
    return circuit.get_all_derivatives(hamiltonian)
    state = circuit()
    #print(np.array([2 * np.dot(state.conj(), hamiltonian(circuit.derivative(i))).real \
    #                 for i in range(len(cur_params))]))
    return np.array([2 * np.dot(state.conj(), hamiltonian(circuit.derivative(i))).real \
                     for i in range(len(cur_params))])

def check_gradients(energy_val, args, circuit = None, \
                    hamiltonian = None, config = None):
    cur_params = circuit.get_parameters()
    grads = get_all_derivatives(cur_params, circuit, hamiltonian, config)

    for i in range(len(grads)):
        new_params = cur_params.copy()
        new_params[i] += 1e-7
        energy_i = _circuit_energy(cur_params, *args)
        energy_f = _circuit_energy(new_params, *args)

        print(i, (energy_f - energy_i) / 1e-7, grads[i])
        assert np.abs((energy_f - energy_i) / 1e-7 - grads[i]) < 1e-5

    return circuit



class Optimizer(object):
    def __init__(self, hamiltonian, circuit, algorithm, config, param_dict):
        self.hamiltonian = hamiltonian
        self.circuit = circuit
        self.algorithm = algorithm
        self.alg_param_dict = param_dict
        self.config = config

        # self.alg_param_dict['bounds'] = [(-np.pi, np.pi)] * len(circuit.get_parameters())
        return

    def optimize(self):
        #check_gradients(_circuit_energy, args=(self.circuit, self.hamiltonian, self.config), hamiltonian = self.hamiltonian, \
        #                circuit = self.circuit, config = self.config)
        res = self.algorithm(_circuit_energy, self.circuit.get_parameters(), \
                             args=(self.circuit, self.hamiltonian, self.config), \
                             jac = get_all_derivatives, **self.alg_param_dict)

        return res.x
