import scipy as sp
import numpy as np
import utils
from time import time
'''
def _circuit_energy(param, circuit, hamiltonian, config, projector):
    circuit.set_parameters(param)
    
    #norm_sample = utils.compute_norm_sample(state, projector, config.N_samples)
    #energy_sample = utils.compute_energy_sample(state, hamiltonian, projector, config.N_samples)
    #energy_sample_symmetrized = utils.compute_energy_sample_symmetrized(state, hamiltonian, projector, config.N_samples)
    #print('norm :', norm, norm_sample)
    #print('energy: ', np.dot(np.conj(state), hamiltonian(state_proj) / norm).real, energy_sample / norm_sample, energy_sample_symmetrized / norm_sample)
    state = circuit()
    if config.N_samples is None:
        state = circuit()
        assert np.isclose(state.conj().dot(state), 1.0)
        state_proj = projector(state)
        energy = np.dot(np.conj(state), hamiltonian(state_proj))
        norm = np.dot(state.conj(), state_proj)
        return (energy / norm).real

    
    energy_sample = utils.compute_energy_sample(state, hamiltonian, projector, config.N_samples)
    norm_sample = utils.compute_norm_sample(state, projector, config.N_samples)
    return (energy_sample / norm_sample).real
'''

def gradiend_descend(energy_val, init_values, args, circuit = None, \
                     hamiltonian = None, config = None, projector = None, \
                     n_iter = 100, lr = 0.003):
    for n_iter in range(n_iter):
        cur_params = circuit.get_parameters()
        state = circuit()
        state_proj = projector(state)
        norm = no.dot(state.conj(), state_proj)

        energy = _circuit_energy(param, circuit, hamiltonian, config, projector)

        grads = []
        for i in range(len(cur_params)):
            der_i = circuit.derivative(i)
            projected_der_i = projector(der_i)
            connectivity_i = np.dot(state.conj(), projected_der_i) / norm

            grad = np.dot(state.conj(), hamiltonian(projected_der_i)) / norm
            grads.append(2 * (grad - connectivity_i * energy).real)

        new_params = (cur_params - lr * grads).real

        circuit.set_parameters(new_params)

        print('iteration: {:d}, energy = {:.7f}, fidelity = {:.6f}'.format(n_iter, energy - 33., np.abs(np.dot(hamiltonian.ground_state[0], state_proj)) ** 2))
        #print(new_params)
    return circuit

def natural_gradiend_descend(obs, init_values, args, n_iter = 2000, lr = 0.003, test = False):
    circuit, hamiltonian, config, projector = args
    lambdas = 3. - np.linspace(0, 2.7, n_iter)

    for n_iter in range(n_iter):
        t_iter = time()
        cur_params = circuit.get_parameters()
        t = time()
        if config.N_samples is None:
            grads_exact, ij_exact, der_one_exact = circuit.get_natural_gradients(hamiltonian, projector, config.N_samples)
        else:
            grads_exact, ij_exact, der_one_exact, grads, ij, der_one = circuit.get_natural_gradients(hamiltonian, projector, config.N_samples)
        print('get all gradients and M_ij', time() - t)
        #print('grads_exact:', grads_exact)
        #print('grads_sampled:', grads)

        #print('connectivity exact', der_one_exact)
        #print('connectivity sampled', der_one)


        #print('connectivities')
        #for conexact, consampl in zip(der_one_exact, der_one):
        #    print(conexact, consampl)

        #np.save('conn_exact.npy', der_one_exact)
        #np.save('conn_sampl.npy', der_one)

        #print('grads')
        #for conexact, consampl in zip(grads_exact, grads):
        #    print(conexact, consampl)

        #np.save('grads_exact.npy', grads_exact)
        #np.save('grads_sampl.npy', grads)


        #for i in range(ij.shape[0]):
        #    for j in range(ij.shape[1]):
        #        print(i, j, ij_exact[i, j], ij[i, j])

        #np.save('MT_exact.npy', ij_exact)
        #np.save('MT_sampl.npy', ij)

        #print('ij discrepancy:',  np.linalg.norm(ij - ij_exact) / np.linalg.norm(ij_exact))
        if config.N_samples is not None:
            MT = (ij - np.einsum('i,j->ij', der_one.conj(), der_one)).real
        if config.test or config.N_samples is None:
            MT_exact = (ij_exact - np.einsum('i,j->ij', der_one_exact.conj(), der_one_exact)).real

        #print('MT discrepancy:',  np.linalg.norm(MT - MT_exact) / np.linalg.norm(MT_exact))

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
        if config.N_samples is not None:
            MT = (ij - np.einsum('i,j->ij', der_one.conj(), der_one)).real
            MT += config.SR_diag_reg * np.diag(np.diag(MT))
            #assert np.allclose(MT, MT.T)


            s, u = np.linalg.eigh(MT)
            #print('s sampled:', s)
            #print('u sampled:', u.T)
            MT_inv = np.zeros(MT.shape)
            keep_lambdas = (s / s.max()) > config.SR_eig_cut
            for lambda_idx in range(len(s)):
                if not keep_lambdas[lambda_idx]:
                    continue
                MT_inv += (1. / s[lambda_idx]) * \
                        np.einsum('i,j->ij', u[:, lambda_idx], u[:, lambda_idx])

            circuit.forces = grads.copy()
            grads = MT_inv.dot(grads - lambdas[n_iter] * der_one.real)
            circuit.forces_SR = grads.copy()

        if config.test or config.N_samples is None:
            MT_exact = (ij_exact - np.einsum('i,j->ij', der_one_exact.conj(), der_one_exact)).real
            MT_exact += config.SR_diag_reg * np.diag(np.diag(MT_exact))

            #assert np.allclose(MT_exact, MT_exact.T)

            s, u = np.linalg.eigh(MT_exact)
            #print('s exact:', s)
            #print('u exact:', u.T)

            MTe_inv = np.zeros(MT_exact.shape)
            keep_lambdas = (s / s.max()) > config.SR_eig_cut
            #print(keep_lambdas)
            #print(u)
            for lambda_idx in range(len(s)):
                if not keep_lambdas[lambda_idx]:
                    continue
                MTe_inv += (1. / s[lambda_idx]) * \
                          np.einsum('i,j->ij', u[:, lambda_idx], u[:, lambda_idx])
            #assert np.allclose(MTe_inv, np.linalg.inv(MT_exact))

            circuit.forces_exact = grads_exact.copy()
            grads_exact = MTe_inv.dot(grads_exact)
            circuit.forces_SR_exact = grads_exact.copy()
            #if np.sum(np.abs(grads)) / len(grads) > 3:
            #    print('flipped')
            #    grads = 3 * grads / np.sqrt(np.sum(grads ** 2))

        
            #print('grads SR')
            #for conexact, consampl in zip(grads_exact, grads):
            #    print(conexact, consampl)

            #np.save('gradsSR_exact.npy', grads_exact)
            #np.save('gradsSR_sampl.npy', grads)

            #np.save('MT_inv_exact.npy', MTe_inv)
            #np.save('MT_inv_sampl.npy', MT_inv)

            #exit(-1)
        if config.N_samples is not None:
            new_params = (cur_params - lr * grads).real
        else:
            new_params = (cur_params - lr * grads_exact).real
                   #print('forces_sampled =', repr(grads))
            #print('forces_exact =', repr(grads_exact))
            #print('current parameters =', repr(new_params))


        circuit.set_parameters(new_params)

        obs.write_logs()
        #state = circuit()
        #assert np.isclose(state.conj().dot(state), 1.0)
        #state_proj = projector(state)
        #state_proj = state_proj / np.sqrt(np.dot(state_proj.conj(), state_proj))

        #print('iteration: {:d}, energy = {:.7f}, fidelity = {:.7f}'.format(n_iter, _circuit_energy(new_params, *args) - hamiltonian.energy_renorm, \
        #                np.abs(np.dot(hamiltonian.ground_state[0].conj(), state_proj)) ** 2))
        print('iteration took', time() - t_iter)
    return circuit

def get_all_derivatives(cur_params, circuit, hamiltonian, config, projector):
    return circuit.get_all_derivatives(hamiltonian, projector)

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
    def __init__(self, hamiltonian, circuit, projector, obs, algorithm, config, param_dict):
        self.hamiltonian = hamiltonian
        self.circuit = circuit
        self.algorithm = algorithm
        self.projector = projector
        self.alg_param_dict = param_dict
        self.config = config
        self.obs = obs

        return

    def optimize(self):
        #check_gradients(_circuit_energy, args=(self.circuit, self.hamiltonian, self.config), hamiltonian = self.hamiltonian, \
        #                circuit = self.circuit, config = self.config)
        #res = self.algorithm(_circuit_energy, self.circuit.get_parameters(), \
        #                     args=(self.circuit, self.hamiltonian, self.config, self.projector), \
        #                     jac = get_all_derivatives, **self.alg_param_dict)

        res = self.algorithm(self.obs, self.circuit.get_parameters(), \
                             args=(self.circuit, self.hamiltonian, self.config, self.projector), \
                             **self.alg_param_dict)

        return res.x
