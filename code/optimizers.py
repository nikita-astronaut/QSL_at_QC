import scipy as sp
import numpy as np
import utils

def _circuit_energy(param, circuit, hamiltonian, config):
    circuit.set_parameters(param)
    state = circuit()
    #print('norm', state.conj().dot(state))
    assert np.isclose(state.conj().dot(state), 1.0)
    
    #for i in range(len(state)):
    #    print(utils.index_to_spin(np.array([i]), number_spins = 16), state[i])
    print(np.dot(np.conj(state), hamiltonian(state)).real, flush = True)
    #print(param)
    return np.dot(np.conj(state), hamiltonian(state)).real


def gradiend_descend(energy_val, init_values, args, circuit = None, \
                     hamiltonian = None, config = None, n_iter = 100, lr = 0.01):
    for n_iter in range(n_iter):
        cur_params = circuit.get_parameters()
        state = circuit()

        grads = np.array([2 * np.dot(state.conj(), hamiltonian(circuit.derivative(i))).real \
                          for i in range(len(cur_params))])
        new_params = (cur_params - lr * grads).real

        circuit.set_parameters(new_params)

        print('iteration: {:d}, energy = {:.7f}'.format(n_iter, _circuit_energy(new_params, *args)))
        #print(new_params)
    return circuit

def natural_gradiend_descend(energy_val, init_values, args, n_iter = 10000, lr = 0.01, test = False):
    circuit, hamiltonian, config = args
    for n_iter in range(n_iter):
        cur_params = circuit.get_parameters()
        grads, ij, der_one = circuit.get_natural_gradients(hamiltonian)
                
        if test:
            for i in range(len(grads)):
                state_i = circuit()
                new_params = cur_params.copy()
                new_params[i] += 1e-9
                circuit.set_parameters(new_params)
                state_f = circuit()
                der = np.dot(state_i.conj(), state_f - state_i) / 1e-9

                print(der_one[i], der, i)
                # assert np.isclose(der_one[i], der)
                circuit.set_parameters(cur_params)

            for i in range(len(grads)):
                for k in range(len(grads)):
                    state_0 = circuit()
                    new_params = cur_params.copy()

                    new_params[i] += 1e-6
                    circuit.set_parameters(new_params)
                    state_i = circuit()

                    new_params[i] -= 1e-6  
                    new_params[k] += 1e-6  
                    circuit.set_parameters(new_params)
                    state_k = circuit()
                    circuit.set_parameters(cur_params)

                    der = np.dot((state_i - state_0).conj(), state_k - state_0) / 1e-6 / 1e-6
                    print(ij[i, k], der, i, k)
                    assert np.abs((ij[i, k] - der)) < 1e-3

            #print(j[i], der)
            #assert np.isclose(j[i], der)
            #circuit.set_parameters(cur_params)
        

        #circuit.set_parameters(cur_params)
        MT = (ij - np.einsum('i,j->ij', der_one.conj(), der_one)).real
        grads = np.linalg.inv(MT + 1e-8 * np.eye(MT.shape[0])).dot(grads)
        #if np.sum(np.abs(grads)) / len(grads) > 3:
        #    print('flipped')
        #    grads = 3 * grads / np.sqrt(np.sum(grads ** 2))


        new_params = (cur_params - lr * grads).real
        
        print('forces =', grads)
        print('current parameters =', new_params)

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
        assert np.abs((energy_f - energy_i) / 1e-7 - grads[i]) < 1e-3

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

        #res = self.algorithm(_circuit_energy, self.circuit.get_parameters(), \
        #                     args=(self.circuit, self.hamiltonian, self.config), \
        #                     **self.alg_param_dict)

        return res.x
