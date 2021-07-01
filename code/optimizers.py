import scipy as sp
import numpy as np
import utils
from time import time

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

def natural_gradiend_descend(obs, init_values, args, n_iter = 40000, lr = 0.003, test = False):
    circuit, hamiltonian, config, projector = args

    #lambdas = 0.1 * np.concatenate([\
    #                              np.ones(2000) * 3., \
    #                              np.ones(1000) * 1., \
    #                              np.ones(1000) * 0.3 \
    #                             ])
    lambdas = 1. * np.ones(100000)


    energies = []
    parameters = []


    circuit.lamb = 0.
    for n_iter in range(n_iter):
        t_iter = time()
        cur_params = circuit.get_parameters()
        t = time()
        if config.N_samples is None:
            grads_exact, ij_exact, der_one_exact = circuit.get_natural_gradients(hamiltonian, projector, config.N_samples)
        else:
            grads_exact, ij_exact, der_one_exact, grads, ij, der_one = circuit.get_natural_gradients(hamiltonian, projector, config.N_samples)
        print('get all gradients and M_ij', time() - t)
        print(np.linalg.eigh(ij)[0])
        exit(-1)
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

        if config.test:
            print('grads')
            for conexact, consampl in zip(grads_exact, grads):
                print(conexact, consampl, np.abs(conexact - consampl))

            print('connectivities')
            for conexact, consampl in zip(der_one_exact, der_one):
                print(conexact, consampl, np.abs(conexact - consampl))

        #np.save('grads_exact.npy', grads_exact)
        #np.save('grads_sampl.npy', grads)

        if config.test:
            for i in range(ij.shape[0]):
                for j in range(ij.shape[1]):
                    if i == j:
                        print(i, j, ij_exact[i, j], ij[i, j])

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


            if config.reg == 'svd':
                s, u = np.linalg.eigh(MT)
                MT_inv = np.zeros(MT.shape)
                keep_lambdas = (s / s.max()) > config.SR_eig_cut
                for lambda_idx in range(len(s)):
                    if not keep_lambdas[lambda_idx]:
                        continue
                    MT_inv += (1. / s[lambda_idx]) * \
                            np.einsum('i,j->ij', u[:, lambda_idx], u[:, lambda_idx])
            else:
                MT2 = MT @ MT.T.conj()
                eigvals, eigstates = np.linalg.eigh(MT2)
                assert np.all(eigvals > 0)
                MT = np.einsum('i,ij,ik->jk', np.sqrt(eigvals), eigstates.T, eigstates.T.conj()) + config.SR_eig_cut * np.eye(MT.shape[0])
                MT_inv = np.linalg.inv(MT)
                #MT_inv = np.linalg.inv(MT + config.SR_eig_cut * np.eye(MT.shape[0]))

            circuit.forces = grads.copy()
            #grads = MT_inv.dot(grads - lambdas[n_iter] * der_one.real)
            grads = MT_inv.dot(grads - circuit.lamb * der_one.real * (1. if config.lagrange else 0.))  # FIXME: shall we include this to the SR?
            circuit.forces_SR = grads.copy()

        if config.test or config.N_samples is None:
            MT_exact = (ij_exact - np.einsum('i,j->ij', der_one_exact.conj(), der_one_exact)).real
            MT_exact += config.SR_diag_reg * np.diag(np.diag(MT_exact))

            #assert np.allclose(MT_exact, MT_exact.T)

            if config.reg == 'svd':
                s, u = np.linalg.eigh(MT_exact)

                MTe_inv = np.zeros(MT_exact.shape)
                keep_lambdas = (s / s.max()) > config.SR_eig_cut
                for lambda_idx in range(len(s)):
                    if not keep_lambdas[lambda_idx]:
                        continue
                    MTe_inv += (1. / s[lambda_idx]) * \
                              np.einsum('i,j->ij', u[:, lambda_idx], u[:, lambda_idx])
            else:
                MTe_inv = np.linalg.inv(MTe + config.SR_eig_cut * np.eye(MTe.shape[0]))
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
            if config.lagrange:
                circuit.lamb -= (circuit.norm - config.target_norm) * config.Z * lr
        else:
            new_params = (cur_params - lr * grads_exact).real
        if config.test:
            print('forces_sampled =', repr(grads))
            print('forces_exact =', repr(grads_exact))
            #print('current parameters =', repr(new_params))

        if config.max_energy_increase_threshold is None:
            circuit.set_parameters(new_params)
        else:

            if len(energies) > 0 and circuit.energy > energies[-1] + config.max_energy_increase_threshold:
                circuit.set_parameters(parameters[-1])
                print('energy increase over threshold: from', energies[-1] - hamiltonian.energy_renorm, ' to ', circuit.energy - hamiltonian.energy_renorm)
            else:
                circuit.set_parameters(new_params)

        energies.append(circuit.energy)
        parameters.append(circuit.get_parameters())


        #circuit.set_parameters(new_params)

        obs.write_logs()
        #state = circuit()
        #assert np.isclose(state.conj().dot(state), 1.0)
        #state_proj = projector(state)
        #state_proj = state_proj / np.sqrt(np.dot(state_proj.conj(), state_proj))

        #print('iteration: {:d}, energy = {:.7f}, fidelity = {:.7f}'.format(n_iter, _circuit_energy(new_params, *args) - hamiltonian.energy_renorm, \
        #                np.abs(np.dot(hamiltonian.ground_state[0].conj(), state_proj)) ** 2))
        print('iteration took', time() - t_iter)
        print('lambda = {:.3f}'.format(circuit.lamb))
    return circuit



def SPSA_gradiend_descend(obs, init_values, args, n_iter = 40000, lr = 0.003, test = False):
    circuit, hamiltonian, config, projector = args

    lambdas = 1. * np.ones(100000)
    MT_smoothed = np.eye(len(circuit.params))

    lamb = 0.
    energies = []
    parameters = []

    ## ADAM parameters ##
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-8

    v = None
    m = None

    for n_iter in range(n_iter):
        t_iter = time()
        cur_params = circuit.get_parameters()
        #circuit.fix_noise_model_SPSA()
        t = time()
        
        grads_exact, ij_exact, der_one_exact, grads, ij, der_one = circuit.get_natural_gradients(hamiltonian, projector, config.N_samples, 'SPSA')
        print('get all gradients and M_ij', time() - t)


        
        #MT_exact = (ij_exact - np.einsum('i,j->ij', der_one_exact.conj(), der_one_exact)).real

        print('eigenvalues_SPSA', np.linalg.eigh(ij)[0])
        #print('eigenvalues_exact', np.linalg.eigh(MT_exact)[0])

        #print('grads SPSA', grads)
        #print('grads exact', grads_exact)

        MT_smoothed = MT_smoothed * (n_iter + 1) / (n_iter + 2) + ij / (n_iter + 2)
        MT2 = MT_smoothed @ MT_smoothed.T.conj()
        eigvals, eigstates = np.linalg.eigh(MT2)
        #assert np.all(eigvals > 0)
        print(eigvals)
        MT = np.einsum('i,ij,ik->jk', np.sqrt(np.abs(eigvals)), eigstates.T, eigstates.T.conj()) + config.SR_eig_cut * np.eye(MT2.shape[0])

        print(np.linalg.eigh(MT)[0])
        #assert np.all(np.linalg.eigh(MT)[0] > 0)
        MT_inv = np.linalg.inv(MT)

        print('eigenvalues_regularized_SPSA_inv', np.linalg.eigh(MT_inv)[0])

        '''
        ### START DEBUG ###
        MT_exact = (ij_exact - np.einsum('i,j->ij', der_one_exact.conj(), der_one_exact)).real
        MT2 = MT_exact @ MT_exact.T.conj()
        eigvals, eigstates = np.linalg.eigh(MT2)
        assert np.all(eigvals > 0)
        MT_exact = np.einsum('i,ij,ik->jk', np.sqrt(eigvals), eigstates.T, eigstates.T.conj()) + config.SR_eig_cut * np.eye(MT2.shape[0])

        #assert np.all(np.linalg.eigh(MT)[0] > 0)
        MT_exact_inv = np.linalg.inv(MT_exact)

        print('eigenvalues_regularized_exact_inv', np.linalg.eigh(MT_exact_inv)[0])


        exit(-1)
        '''

        ### END DEBUG ###

        circuit.forces = grads.copy()

        #grads = MT_inv.dot(grads - circuit.lamb * der_one.real * (1. if config.lagrange else 0.))
        #circuit.forces_SR = grads.copy()

        m = grads if m is None else m * beta1 + (1 - beta1) * grads
        v = np.vdot(grads, grads) if v is None else v * beta2 + (1. - beta2) * np.vdot(grads, grads)

        #grads = m / (np.sqrt(v) + epsilon)
        grads = MT_inv.dot(grads - circuit.lamb * der_one.real * (1. if config.lagrange else 0.))
        circuit.forces_SR = grads.copy()

        new_params = (cur_params - lr * grads).real
        if config.lagrange:
            circuit.lamb -= (circuit.norm - config.target_norm) * config.Z * lr


        if len(energies) > 0 and circuit.energy > energies[-1] + config.max_energy_increase_threshold:
            circuit.set_parameters(parameters[-1])
            print('energy increase over threshold: from', energies[-1] - hamiltonian.energy_renorm, ' to ', circuit.energy - hamiltonian.energy_renorm)
        else:
            circuit.set_parameters(new_params)

        energies.append(circuit.energy)
        parameters.append(circuit.get_parameters())

        print('iteration took', time() - t_iter)
        print('lambda = {:.3f}'.format(circuit.lamb))
        print('energy from estimation', circuit.energy - hamiltonian.energy_renorm)
        obs.write_logs()
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
