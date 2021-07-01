import numpy as np
from copy import deepcopy
import os
import scipy as sp
from scipy import sparse
import lattice_symmetries as ls
import scipy
import utils
from time import time
import qiskit


sz = np.array([[1, 0], \
               [0, -1]])

sx = np.array(
               [[0, 1], \
               [1, 0]]
              )

sy = np.array([[0, 1.0j], [-1.0j, 0]])

s0 = np.eye(2)

SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
SSun = -np.kron(sx, sx) + -np.kron(sy, sy) + np.kron(sz, sz)
P_ij_global = (SS + np.eye(4)) / 2.



class Circuit(object):
    def __init__(self, n_qubits, basis, config, unitary, **kwargs):
        self.basis = basis
        self.config = config
        self.n_qubits = n_qubits
        self.unitary = unitary
        self.unitary_site = unitary[0, :]
        self.dimerization = config.dimerization

        self.basis_bare = ls.SpinBasis(ls.Group([]), number_spins=n_qubits, hamming_weight=None)
        self.basis_bare.build()

        self.forces_exact = None
        self.forces = None
        self.forces_SR_exact = None
        self.forces_SR = None

    def __call__(self):
        state = self._initial_state()
        for gate in self.unitaries:
            state = gate(state)
        assert np.isclose(np.dot(state, self.total_spin(state)) + 3 * self.Lx * self.Ly, 0.0)
        assert np.isclose(np.dot(state, state), 1.0)
        return state

    def get_natural_gradients(self, hamiltonian, projector, N_samples=None, method='standard'):
        t = time()
        ij, j, ij_sampling, j_sampling = self.get_metric_tensor(projector, N_samples, method)
        print('metric tensor: ', time() - t)
        self.connectivity_sampling = j_sampling

        t = time()
        grads, grads_sampling = self.get_all_derivatives(hamiltonian, projector, N_samples, method)
        print('energy derivatives: ', time() - t)

        if N_samples is None:
            return grads, ij, j
        return grads, ij, j, grads_sampling, ij_sampling, j_sampling


    def _get_derivative_idx(self, param_idx):
        return self.derivatives[param_idx]

    def _initialize_parameters(self):
        return np.random.uniform(size=len(self.locs))

    def get_parameters(self):
        return self._pack_parameters()

    def set_parameters(self, parameters, reduced=False):
        self.params = parameters.copy()
        self._refresh_unitaries_derivatives(reduced=reduced)
        return

    def _unpack_parameters(self, parameters):
        return parameters

    def _pack_parameters(self):
        return self.params

    def _initial_state(self):
        raise NotImplementedError()

    def _refresh_unitaries_derivatives(self):
        self.unitaries = []
        self.unitaries_herm = []
        self.derivatives = []
        for m, loc, par in zip(self.matrices, self.locs, self.params):
            self.unitaries.append(ls.Operator(self.basis, \
                [ls.Interaction(scipy.linalg.expm(1.0j * par * m), [loc])]))
            self.unitaries_herm.append(ls.Operator(self.basis, \
                [ls.Interaction(scipy.linalg.expm(-1.0j * par * m), [loc])]))
            self.derivatives.append(ls.Operator(self.basis, [ls.Interaction(1.0j * m, [loc])]))
        return 

class TrotterizedMarshallsSquareHeisenbergNNAFM(Circuit):
    '''
        constructs unitary circuit
        U(\\theta_i, \\theta_perp_lm, \\theta_Z_lm) = \\prod_i e^{-i \\theta_i Z_i} \\prod_{l < m}  A_lm(\\theta_perp_lm) B_lm(\\thetaZ_lm)
        where B_lm(\\thetaZ_lm) = exp(-i Z_l Z_m \\theta_lm)
              A_lm(\\thetaZ_lm) = exp(-i [Y_l Y_m + X_l X_m] \\theta_lm)
    '''
    def __init__(self, Lx, Ly, n_lm_neighbors=2):
        self.Lx = Lx
        self.Ly = Ly
        self.n_lm_neighbors = n_lm_neighbors
        super().__init__(Lx * Ly)

        self.pairwise_distances = self.get_pairwise_distances()
        self.i_sites, self.pair_bonds = self._define_parameter_site_bonds()


        ### defining operator locs ###
        self.matrices, self.locs = self._get_matrices_locs()
        self.params = self._initialize_parameters()


        ### defining of unitaries ###
        self._refresh_unitaries_derivatives()
        return

    def _define_parameter_site_bonds(self):
        i_sites = np.arange(self.Lx * self.Ly)
        pair_bonds = []

        unique_distances = np.sort(np.unique(self.pairwise_distances.flatten()))


        for l in range(self.Lx * self.Ly):
            for m in range(l + 1, self.Lx * self.Ly):
                dist = self.pairwise_distances[l, m]

                separation = np.where(unique_distances == dist)[0][0]

                assert separation > 0

                if separation >= self.n_lm_neighbors:
                    continue

                pair_bonds.append((l, m))

        return i_sites, pair_bonds

    def get_pairwise_distances(self):
        distances = np.zeros((self.Lx * self.Ly, self.Lx * self.Ly))

        for i in range(self.Lx * self.Ly):
            for j in range(self.Lx * self.Ly):
                xi, yi = i % self.Lx, i // self.Lx
                xj, yj = j % self.Lx, j // self.Lx

                distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)  # OBC
                distances[i, j] = distance

        assert np.allclose(distances, distances.T)

        return np.around(distances, decimals=4)

    def _get_matrices_locs(self):
        matrices = []
        locs = []

        for i in range(len(self.pair_bonds)):
            matrices.append(np.kron(sx, sx) + np.kron(sy, sy))
            matrices.append(np.kron(sz, sz))
            locs.append(self.pair_bonds[i])
            locs.append(self.pair_bonds[i])

        for i in range(len(self.i_sites)):
            locs.append((self.i_sites[i],))
            matrices.append(sz)

        for i in range(len(self.pair_bonds)):
            matrices.append(np.kron(sx, sx) + np.kron(sy, sy))
            matrices.append(np.kron(sz, sz))
            locs.append(self.pair_bonds[i])
            locs.append(self.pair_bonds[i])

        for i in range(len(self.i_sites)):
            locs.append((self.i_sites[i],))
            matrices.append(sz)

        return matrices, locs

    def _initial_state(self):
        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        spin_1 = np.zeros(self.n_qubits, dtype=np.int64)
        spin_2 = np.zeros(self.n_qubits, dtype=np.int64)
        for i in range(self.n_qubits):
            x, y = i % self.Lx, i // self.Lx
            if (x + y) % 2 == 0:
                spin_1[i] = 1
            else:
                spin_2[i] = 1


        state[utils.spin_to_index(spin_1, number_spins = self.n_qubits)] = 1. / np.sqrt(2)
        state[utils.spin_to_index(spin_2, number_spins = self.n_qubits)] = -1. / np.sqrt(2)
        return state


class SU2_symmetrized(Circuit):
    def __init__(self, subl, Lx, Ly, basis, config, unitary, BC, spin=0):
        self.BC = BC
        self.Lx = Lx
        self.Ly = Ly
        self.n_subl = subl
        self.n_qubits = Lx * Ly * subl
        self.spin = spin
        self.dimerization = config.dimerization
        self.lamb = 10.
        super().__init__(Lx * Ly * subl, basis, config, unitary)

        # init initial state of the circuit
        all_bonds = []
        all_bondsun = []
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i == j:
                    continue
                if self.unitary[i, j] == +1:
                    all_bonds.append((i, j))
                else:
                    all_bondsun.append((i, j))
        self.total_spin = ls.Operator(self.basis, ([ls.Interaction(SS, all_bonds)] if len(all_bonds) > 0 else []) + \
                                                  ([ls.Interaction(SSun, all_bondsun)] if len(all_bondsun) > 0 else []))


        self.singletizer = np.zeros((4, 4), dtype=np.complex128)
        self.singletizer[1, 0] = 1. / np.sqrt(2)
        self.singletizer[2, 0] = -1. / np.sqrt(2)
        self.singletizer = self.singletizer + self.singletizer.T

        self.tripletizer = np.zeros((4, 4), dtype=np.complex128)
        self.tripletizer[3, 0] = 1.
        self.tripletizer = self.tripletizer + self.tripletizer.T

        self.octupletizer = np.zeros((16, 16), dtype=np.complex128)
        self.octupletizer[15, 0] = 1.

        self.octupletizer = self.octupletizer + self.octupletizer.T


        self.ini_state = None
        self.ini_state = self._initial_state()

        ### defining operator locs ###
        self.layers, self.pairs = self._get_dimerizarion_layers()
        self.params = self._initialize_parameters()


        ### defining of unitaries ###
        self._refresh_unitaries_derivatives()


        ### defining the noise ###
        self.noise_operators = []
        for ai, si in enumerate([s0, sx, sy, sz]):
            for aj, sj in enumerate([s0, sx, sy, sz]):
                if ai + aj == 0:
                    continue
                self.noise_operators.append(np.kron(si, sj))

        return

    def __call__(self, noisy=None):
        if (noisy is None and self.config.noise) or noisy:
            state = self._initial_state_noisy()
            apply_noise = np.random.uniform(0, 1, size=len(self.params)) < self.config.noise_p
            which_noise = np.floor(np.random.uniform(0, 1, size=len(self.params)) * 15).astype(np.int64)
            ctr = 0
            for layer in self.unitaries:
                for gate in layer:
                    state = gate(state)

                    if apply_noise[ctr]:
                        print('applied noise on pair', self.pairs[ctr], which_noise[ctr])
                        op = ls.Operator(self.basis, [ls.Interaction(self.noise_operators[which_noise[ctr]], [self.pairs[ctr]])])
                        state = op(state)

                    ctr += 1


        state = self._initial_state()
        for layer in self.unitaries:
            for gate in layer:
                state = gate(state)
        return state



    def get_all_derivatives(self, hamiltonian, projector, N_samples, method='standard'):
        t = time()

        if N_samples is None or self.config.test == True:
            state = self.__call__()
            state_proj = projector(state)
            norm = np.dot(state.conj(), state_proj)

            energy = np.dot(np.conj(state), hamiltonian(state_proj) / norm)

            LEFT = hamiltonian(state_proj)
            LEFT_conn = state_proj.copy()  # total projector is Hermitean

            for layer in reversed(self.unitaries_herm):
                for u_herm in reversed(layer):
                    LEFT = u_herm(LEFT)
                    LEFT_conn = u_herm(LEFT_conn)
            LEFT = LEFT.conj()
            LEFT_conn = LEFT_conn.conj()

            RIGHT = self._initial_state()


            grads = []
            numerators = []
            numerators_conn = []
            for idx, layer in enumerate(self.unitaries):
                derivative = RIGHT * 0.0
                for der in self.derivatives[idx]:
                    derivative += der(RIGHT)

                grad = np.dot(LEFT, derivative) / norm
                numerators.append(np.dot(LEFT, derivative))
                grad -= np.dot(LEFT_conn, derivative) / norm * energy
                numerators_conn.append(np.dot(LEFT_conn, derivative))
                grads.append(2 * grad.real)


                LEFT = LEFT.conj()
                LEFT_conn = LEFT_conn.conj()
                for u in self.unitaries[idx]:
                    RIGHT = u(RIGHT)
                    LEFT = u(LEFT)
                    LEFT_conn = u(LEFT_conn)
                LEFT = LEFT.conj()
                LEFT_conn = LEFT_conn.conj()

            print('grads exact', time() - t)
            if N_samples is None:
                return np.array(grads), None

        
        if method == 'standard':
            derivatives_sampling = utils.compute_energy_der_sample(self.__call__(), self.der_states, hamiltonian, projector, N_samples, self.config.noise_p)
            norm_sampling = utils.compute_norm_sample(self.__call__(), projector, N_samples, self.config.noise_p)
            energy_sampling = utils.compute_energy_sample(self.__call__(), hamiltonian, projector, N_samples, self.config.noise_p)
            grad_sampling = (derivatives_sampling / norm_sampling - self.connectivity_sampling * energy_sampling / norm_sampling).real * 2.

            self.energy = energy_sampling / norm_sampling

            if self.config.qiskit:
                t = time()
                norm_qiskit_ht = utils.compute_norm_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model)
                print('norm_hadamardtest fromwf', time() - t)

                t = time()
                energy_qiskit = []
                for repetition in range(self.config.n_noise_repetitions):
                    energy_qiskit.append(utils.compute_energy_qiskit_hadamardtest(self, projector, hamiltonian, self.config.N_samples // self.config.n_noise_repetitions, self.config.noise_model))
                print('energies:',  energy_qiskit)
                energy_qiskit = np.mean(energy_qiskit)
                print('energy_hadamardtest fromwf', time() - t)

                print('energy', energy_sampling, energy_qiskit)
                print('norm', norm_sampling, norm_qiskit_ht)



        elif method == 'SPSA':
            norm_sampling = 1.
            derivatives_sampling = None
            energy_sampling = None

            
            deltas = []
            outers = []
            for _ in range(self.config.SPSA_gradient_averages):
                delta = (np.random.randint(0, 2, size=len(self.params)) * 2. - 1.) / np.sqrt(len(self.params))
                outers.append(delta)

                deltas.append(delta * 0.)
                deltas.append(delta)
                deltas.append(-delta)
            deltas = np.array(deltas)


            if not self.config.qiskit:
                all_states = self.apply_noisy_circuit_batched(deltas)
            else:
                cur_params = self.params.copy()


            self.energy = 0.
            self.norm = 0.
            grad_sampling = np.zeros((len(self.params)))
            for k in range(self.config.SPSA_gradient_averages):
                if not self.config.qiskit:
                    state_00 = all_states[3 * k]
                    state_p = all_states[3 * k + 1]
                    state_m = all_states[3 * k + 2]

                    Ep = utils.compute_energy_sample(state_p, hamiltonian, projector, N_samples, self.config.noise_p) / utils.compute_norm_sample(state_p, projector, N_samples, self.config.noise_p)
                    Em = utils.compute_energy_sample(state_m, hamiltonian, projector, N_samples, self.config.noise_p) / utils.compute_norm_sample(state_m, projector, N_samples, self.config.noise_p)

                    print(Ep, Em, (Ep - Em) / 2 / self.config.SPSA_epsilon, 'energy diff estimation')
                    self.energy += utils.compute_energy_sample(state_00, hamiltonian, projector, N_samples, self.config.noise_p) / utils.compute_norm_sample(state_00, projector, N_samples, self.config.noise_p)
                    grad_sampling += (Ep - Em) / 2 / self.config.SPSA_epsilon * outers[k]



                else:
                    new_params = cur_params + self.config.SPSA_epsilon * outers[k]
                    self.set_parameters(new_params, reduced=True)

                    energy_qiskit = []
                    for repetition in range(self.config.n_noise_repetitions):
                        energy_qiskit.append(utils.compute_energy_qiskit_hadamardtest(self, projector, hamiltonian, self.config.N_samples // self.config.n_noise_repetitions, self.config.noise_model))
                    Ep = np.mean(energy_qiskit)
                    Np = np.mean([utils.compute_norm_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model) \
                                      for repetition in range(self.config.n_noise_repetitions)])

                    statep = self.__call__()


                    new_params = cur_params - self.config.SPSA_epsilon * outers[k]
                    self.set_parameters(new_params, reduced=True)

                    energy_qiskit = []
                    for repetition in range(self.config.n_noise_repetitions):
                        energy_qiskit.append(utils.compute_energy_qiskit_hadamardtest(self, projector, hamiltonian, self.config.N_samples // self.config.n_noise_repetitions, self.config.noise_model))
                    Em = np.mean(energy_qiskit)
                    Nm = np.mean([utils.compute_norm_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model) \
                                      for repetition in range(self.config.n_noise_repetitions)])

                    statem = self.__call__()

                    Ep_exact = np.vdot(statep, hamiltonian(statep))
                    Em_exact = np.vdot(statem, hamiltonian(statem))

                    Np_exact = np.vdot(statep, projector(statep))
                    Nm_exact = np.vdot(statem, projector(statem))

                    self.energy += (Ep / Np + Em / Nm) / 2.  # biased estimate but fine
                    grad_sampling += (Ep / Np - Em / Nm) / 2 / self.config.SPSA_epsilon * outers[k]
                    print(Ep, Ep_exact)
                    print(Em, Em_exact)
                    print(Np, Np_exact)
                    print((Ep / Np - Em / Nm) / 2 / self.config.SPSA_epsilon)
                    print('exact energy grad', (Ep_exact / Np_exact - Em_exact / Nm_exact) / 2 / self.config.SPSA_epsilon)
                    self.norm += (Np + Nm) / 2.

                    self.set_parameters(cur_params)


                    

            self.norm /= self.config.SPSA_gradient_averages
            grad_sampling /= self.config.SPSA_gradient_averages
            self.energy /= self.config.SPSA_gradient_averages

        if self.config.test:
            return np.array(grads), grad_sampling

        return None, grad_sampling



    def apply_noisy_circuit_batched(self, deltas):
        '''
            deltas -- np.array n_states x n_parameters
        '''
        #states = np.array([self._initial_state_noisy() for _ in range(deltas.shape[0])])  # TODO: speed me up
        states = self._initial_state_noisy_batched(deltas.shape[0])


        '''
        apply_noise = np.random.uniform(0, 1, size=(len(self.params), deltas.shape[0])) < self.config.noise_p
        which_noise = np.floor(np.random.uniform(0, 1, size=(len(self.params), deltas.shape[0])) * 15).astype(np.int64)
        ctr = 0
        for layer in self.unitaries:
            for gate in layer:
                states = gate(states.T).T

                all_idxs = []
                for shift in [-2, -1, 0, 1, 2]:
                    idxs = np.where(np.abs(deltas[:, ctr] * np.sqrt(len(self.params)) - shift) < 1e-5)[0]
                    all_idxs += list(idxs)

                    states[idxs] = ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(1.0j * shift * self.config.SPSA_epsilon * P_ij_global), [self.pairs[ctr]])])(states[idxs].T).T

                assert len(all_idxs) == deltas.shape[0]


                for state_idx in range(deltas.shape[0]):
                    if apply_noise[ctr, state_idx]:
                        print('applied noise', self.pairs[ctr], which_noise[ctr, state_idx])
                        op = ls.Operator(self.basis, [ls.Interaction(self.noise_operators[which_noise[ctr, state_idx]], [self.pairs[ctr]])])
                        states[state_idx] = op(states[state_idx])

                ctr += 1
        '''

        ctr = 0
        for layer in self.unitaries:
            for gate in layer:
                states = gate(states.T).T

                all_idxs = []
                for shift in [-2, -1, 0, 1, 2]:
                    idxs = np.where(np.abs(deltas[:, ctr] * np.sqrt(len(self.params)) - shift) < 1e-5)[0]
                    all_idxs += list(idxs)

                    if shift != 0:
                        states[idxs] = ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(1.0j * shift * self.config.SPSA_epsilon * P_ij_global), [self.pairs[ctr]])])(states[idxs].T).T

                assert len(all_idxs) == deltas.shape[0]


                if self.apply_noise_unit[ctr]:
                    print('applied noise', self.pairs[ctr])
                    op = ls.Operator(self.basis, [ls.Interaction(self.noise_operators[self.which_noise_unit[ctr]], [self.pairs[ctr]])])
                    states = op(states.T).T

                ctr += 1

        return states



    def get_metric_tensor(self, projector, N_samples, method):
        if self.config.test or N_samples is None:
            t = time()
            MT = np.zeros((len(self.params), len(self.params)), dtype=np.complex128)

            left_beforeder = self._initial_state()

            for i in range(len(self.params)):
                if i > 0:
                    for layer in self.unitaries[i - 1:i]:
                        for u in layer:
                            left_beforeder = u(left_beforeder)  # L_i ... L_0 |0>
            
                LEFT = left_beforeder.copy()
                derivative = LEFT * 0.0
                for der in self.derivatives[i]:
                    derivative += der(LEFT)
                LEFT = derivative  # A_i L_{i-1} ... L_0 |0>

                for layer in self.unitaries[i:]:
                    for u in layer:
                        LEFT = u(LEFT) # L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>

                LEFT = projector(LEFT) # P L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>

                for layer in reversed(self.unitaries_herm):
                    for u_herm in reversed(layer):
                        LEFT = u_herm(LEFT) # LEFT = L^+_0 L^+_1 ... L^+_{N - 1} P L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>



                RIGHT = self._initial_state()  # RIGHT = |0>
                for j in range(i):
                    for u in self.unitaries[j]:
                        RIGHT = u(RIGHT)
                        LEFT = u(LEFT)


                for j in range(i, len(self.params)):
                    derivative = RIGHT * 0.
                    for der in self.derivatives[j]:
                        derivative += der(RIGHT)

                    MT[i, j] = np.dot(LEFT.conj(), derivative)
                    MT[j, i] = np.conj(MT[i, j])

                    for u in self.unitaries[j]:
                        RIGHT = u(RIGHT) # LEFT = L^+_{j + 1} ... L^+_{N - 1} P L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>
                        LEFT = u(LEFT) # RIGHT = L_j ... L_0 |0>


            der_i = np.zeros(len(self.params), dtype=np.complex128)
            LEFT = self.__call__()  # L_{N-1} ... L_0 |0>
            LEFT = projector(LEFT)  # P L_{N-1} ... L_0 |0>

            norm = np.dot(self.__call__().conj(), LEFT)  # <psi|P|psi>
            MT = MT / norm

            for layer in reversed(self.unitaries_herm):
                for u_herm in reversed(layer):
                    LEFT = u_herm(LEFT) # LEFT = L^+_0 L^+_1 ... L^+_{N - 1} P L_{N - 1} .. L_i L_{i-1} ... L_0 |0>

            RIGHT = self._initial_state()   # RIGHT = |0>
            for i in range(len(self.params)):
                derivative = RIGHT * 0. + 0.0j
                for der in self.derivatives[i]:
                    derivative += der(RIGHT)

                der_i[i] = np.dot(LEFT.conj(), derivative) / norm  # <psi|P|di psi>
    
                for u in self.unitaries[i]:
                    RIGHT = u(RIGHT) # LEFT = L^+_{i + 1} ... L^+_{N - 1} P L_{N - 1} ... L_0 |0>
                    LEFT = u(LEFT) # RIGHT = L_i ... L_0 |0>

            print('MT exact', time() - t)
            if N_samples is None:
                return MT, der_i, None, None



        if method == 'standard':
            MT_sample, connectivity = self.get_metric_tensor_sampling(projector, N_samples)
            norm_sampling = utils.compute_norm_sample(self.__call__(), projector, N_samples, self.config.noise_p)
            self.norm = norm_sampling
        elif method == 'SPSA':
            MT_sample = np.zeros((len(self.params), len(self.params)), dtype=np.float64)

            deltas = []
            outers = []
            pairs = []
            for _ in range(self.config.SPSA_hessian_averages):
                delta1 = (np.random.randint(0, 2, size=len(self.params)) * 2. - 1.) / np.sqrt(len(self.params))
                delta2 = (np.random.randint(0, 2, size=len(self.params)) * 2. - 1.) / np.sqrt(len(self.params))

                pairs.append((delta1, delta2))

                outers.append(np.outer(delta1, delta2) / 2. + np.outer(delta2, delta1) / 2.)

                deltas.append(delta1 * 0.)
                deltas.append(delta1 + delta2)
                deltas.append(delta1)
                deltas.append(-delta1 + delta2)
                deltas.append(-delta1)
            deltas = np.array(deltas)

            if not self.config.qiskit:
                all_states = self.apply_noisy_circuit_batched(deltas)
            else:
                cur_params = self.params.copy()


            for k in range(self.config.SPSA_hessian_averages):
                if not self.config.qiskit:
                    state_00 = all_states[5 * k]
                    state_p1p2 = all_states[5 * k + 1]
                    state_p1 = all_states[5 * k + 2]
                    state_m1p2 = all_states[5 * k + 3]
                    state_m1 = all_states[5 * k + 4]


                    Fp1p2 = utils.compute_overlap_sample(state_00, state_p1p2, self.config.N_samples)
                    Fp1 = utils.compute_overlap_sample(state_00, state_p1, self.config.N_samples)
                    Fm1p2 = utils.compute_overlap_sample(state_00, state_m1p2, self.config.N_samples)
                    Fm1 = utils.compute_overlap_sample(state_00, state_m1, self.config.N_samples)
                else:
                    state0 = self.__call__()
                    self.set_parameters(cur_params + pairs[k][0] + pairs[k][1])
                    statep1p2 = self.__call__()

                    self.set_parameters(cur_params + pairs[k][0])
                    statep1 = self.__call__()

                    self.set_parameters(cur_params - pairs[k][0] + pairs[k][1])
                    statem1p2 = self.__call__()

                    self.set_parameters(cur_params - pairs[k][0])
                    statem1 = self.__call__()

                    self.set_parameters(cur_params)

                    Fp1p2 = utils.compute_gij_qiskit_survivalrate(self, cur_params + pairs[k][0] + pairs[k][1], N_samples, self.config.noise_model)
                    Fp1 = utils.compute_gij_qiskit_survivalrate(self, cur_params + pairs[k][0], N_samples, self.config.noise_model)
                    Fm1p2 = utils.compute_gij_qiskit_survivalrate(self, cur_params - pairs[k][0] + pairs[k][1], N_samples, self.config.noise_model)
                    Fm1 = utils.compute_gij_qiskit_survivalrate(self, cur_params - pairs[k][0], N_samples, self.config.noise_model)


                    Fp1p2_exact =  np.abs(np.vdot(state0, statep1p2)) ** 2
                    Fp1_exact = np.abs(np.vdot(state0, statep1)) ** 2
                    Fm1p2_exact = np.abs(np.vdot(state0, statem1p2)) ** 2
                    Fm1_exact = np.abs(np.vdot(state0, statem1)) ** 2


                    print('Fp1p2', Fp1p2, Fp1p2_exact)
                    print('Fp1', Fp1, Fp1_exact)
                    print('Fm1p2', Fm1p2, Fm1p2_exact)
                    print('Fm1', Fm1, Fm1_exact)

                print((Fp1p2 - Fp1 - Fm1p2 + Fm1) / 2. / self.config.SPSA_epsilon ** 2)
                print('exact:', (Fp1p2_exact - Fp1_exact - Fm1p2_exact + Fm1_exact) / 2. / self.config.SPSA_epsilon ** 2)

                MT_sample += -(1. / 2.) * (Fp1p2 - Fp1 - Fm1p2 + Fm1) / 2. / self.config.SPSA_epsilon ** 2 * outers[k]
            MT_sample /= self.config.SPSA_hessian_averages


            
            '''
            delta = np.random.randint(0, 2, size=len(self.params)) * 2. - 1.

          
            state_00 = self.__call__()
            cur_params = self.params.copy()
            new_params = cur_params + self.config.SPSA_epsilon * (delta)
            self.set_parameters(new_params)
            state_p = self.__call__()

            new_params = cur_params + self.config.SPSA_epsilon * (-delta)
            self.set_parameters(new_params)
            state_m = self.__call__()

            self.set_parameters(cur_params)

            
            Np = utils.compute_norm_sample(state_p, projector, N_samples, self.config.noise_p)
            Nm = utils.compute_norm_sample(state_m, projector, N_samples, self.config.noise_p)

            connectivity = (Np - Nm) / 2 / self.config.SPSA_epsilon * delta
            '''

            connectivity = 1.  # we need it not
            norm_sampling = 1.
            
            #self.norm = utils.compute_norm_sample(state_00, projector, N_samples, self.config.noise_p)

        ### DEBUG ###        
        if False:#self.config.qiskit:
            #t = time()
            #norm_qiskit_ht = utils.compute_norm_qiskit_hadamardtest_shooting(self, projector, self.config.N_samples, self.config.noise_model)
            #print('norm_hadamardtest shooting', time() - t)


            t = time()
            norm_qiskit_ht = utils.compute_norm_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model)
            print('norm_hadamardtest fromwf', time() - t)

            t = time()
            energy_qiskit = utils.compute_energy_qiskit_hadamardtest(self, projector, hamiltonian, self.config.N_samples, self.config.noise_model)
            print('energy_hadamardtest fromwf', time() - t)
            

            #norm_qiskit_sr = utils.compute_norm_qiskit_survivalrate(self, projector, self.config.N_samples, self.config.noise_model)
            #print('norm_survivalrate', time() - t)

            t = time()
            #connectivity_re = utils.compute_connectivity_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model, real_imag = 'real')
            #connectivity_im = utils.compute_connectivity_qiskit_hadamardtest(self, projector, self.config.N_samples, self.config.noise_model, real_imag = 'imag')

            #print(-connectivity_re + 1.0j * connectivity_im)
            #print(connectivity)
            ### TODO: debug, wrong values ###
            #print('connectivity_hadamardtest', time() - t)

            #def index_to_spin(index, number_spins = 16):
            #    return (((index.reshape(-1, 1) & (1 << np.arange(number_spins)))) > 0)


            #def spin_to_index(spin, number_spins = 16):
            #    a = 2 ** np.arange(number_spins)
            #    return np.einsum('ij,j->i', spin, a)

            #all_confs = np.arange(2 ** self.n_qubits)
            #mapping = spin_to_index(index_to_spin(all_confs)[:, ::-1])
            
            #print(np.vdot(norm_qiskit, projector(self.__call__(), 67, inv=True)))
            print('exact:', np.vdot(projector(self.__call__()), self.__call__()).real, 'statevector sampling:', self.norm, 'qiskit hadtest', norm_qiskit_ht)
        
        ### STOP DEBUG ###



        if self.config.test:
            return MT, der_i, MT_sample / norm_sampling, connectivity / norm_sampling



        ### DEBUG ###
        '''
        save = self.config.noise_p
        self.config.noise_p = 0.
        MT, der_i = self.get_metric_tensor_sampling(projector, N_samples)
        norm_sampling = utils.compute_norm_sample(self.__call__(), projector, N_samples, self.config.noise_p)
        self.config.noise_p = save
        return MT / norm_sampling, der_i / norm_sampling, MT_sample / norm_sampling, connectivity / norm_sampling
        '''
        ### END DEBUG ###
        return None, None, MT_sample / norm_sampling, connectivity / norm_sampling

    def fix_noise_model(self):
        self.apply_noise_unit = np.random.uniform(0, 1, size=len(self.params)) < self.config.noise_p
        self.which_noise_unit = np.floor(np.random.uniform(0, 1, size=len(self.params) * 15)).astype(np.int64)

        self.apply_noise_dimer = np.random.uniform(0, 1, size=len(self.dimerization)) < self.config.noise_p
        self.which_noise_dimer = np.floor(np.random.uniform(0, 1, size=len(self.dimerization) * 15)).astype(np.int64)

        return



    def get_metric_tensor_sampling(self, projector, N_samples):
        t = time()

        '''
        if self.config.noise:
            apply_noise = np.random.uniform(0, 1, size=(len(self.params), len(self.params))) < self.config.noise_p
            which_noise = np.floor(np.random.uniform(0, 1, size=(len(self.params), len(self.params))) * 15)
        '''

        self.der_states = np.asfortranarray(self._initial_state_noisy_batched(len(self.params), fixed=False).T) # np.asfortranarray(np.tile(self._initial_state()[..., np.newaxis], (1, len(self.params))))

        apply_noise = np.random.uniform(0, 1, size=(len(self.params), len(self.params))) < self.config.noise_p
        which_noise = np.floor(np.random.uniform(0, 1, size=(len(self.params), len(self.params))) * 15).astype(np.int64)

        for i in range(len(self.params)):
            self.der_states = self.unitaries[i][0](self.der_states)
            self.der_states[..., i] = self.derivatives[i][0](np.ascontiguousarray(self.der_states[..., i]))

            for idx in range(len(self.params)):
                if apply_noise[idx, i]:
                    print('gates: applied noise on pair', self.pairs[i], 'gate', which_noise[idx, i])
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.noise_operators[which_noise[idx, i]], [self.pairs[i]])])
                    self.der_states[:, idx] = op(self.der_states[:, idx])


            '''
            if self.config.noise and np.any(apply_noise[i]):
                for noise_idx in np.where(apply_noise[i])[0]:
                    pair = self.pairs[noise_idx]
                    noise_type = which_noise[i, noise_idx]

                    op = ls.Operator(self.basis, [ls.Interaction(self.noise_operators[noise_type], [pair])])
                    self.der_states[..., noise_idx] = op(self.der_states[..., noise_idx])
            '''

        
        print('obtain states for MT ', time() - t)
        metric_tensor = utils.compute_metric_tensor_sample(np.array(self.der_states).T, projector, N_samples, self.config.noise_p)

        connectivity = utils.compute_connectivity_sample(self.__call__(), self.der_states.T, projector, N_samples, theta=0., noise_p = self.config.noise_p) + \
                        1.0j * utils.compute_connectivity_sample(self.__call__(), self.der_states.T, projector, N_samples, theta=-1, noise_p = self.config.noise_p)
        return metric_tensor, connectivity


    def _initial_state(self):
        if self.ini_state is not None:
            return self.ini_state.copy()

        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[0] = 1.
       
        if self.spin == 0:
            for pair in self.dimerization:
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                state = op(state)
        elif self.spin == 1:
            for idx, pair in enumerate(self.dimerization):
                if idx != 0:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                else:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.tripletizer, [pair])])
                state = op(state)
        else:
            for idx, pair in enumerate(self.dimerization):
                if idx == len(self.dimerization) - 1:
                    break
                if idx < len(self.dimerization) - 2:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                else:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.octupletizer, [tuple(list(self.dimerization[-2]) + list(self.dimerization[-1]))])])
                state = op(state)

        assert np.isclose(np.dot(state.conj(), state), 1.0)
        
        ### perform the unitary transform ###
        for site, phase in enumerate(self.unitary_site):
            op = ls.Operator(self.basis_bare, [ls.Interaction(sz, [(site,)])])
            if phase == -1:
                state = op(state)

        state_su2 = np.zeros(self.basis.number_states, dtype=np.complex128)
        for i in range(self.basis.number_states):
            x = self.basis.states[i]
            _, _, norm = self.basis.state_info(x)
            state_su2[i] = state[self.basis_bare.index(x)] / norm


        assert np.isclose(np.dot(state_su2.conj(), state_su2), 1.0)
        assert np.isclose(np.dot(state_su2.conj(), self.total_spin(state_su2)) + 3. * self.n_qubits, self.spin * (self.spin + 1) * 4.)
        
        return state_su2



    def _initial_state_noisy(self):
        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[0] = 1.

        apply_noise = np.random.uniform(0, 1, len(self.dimerization)) < self.config.noise_p
        which_noise = np.floor(np.random.uniform(0, 1, size=len(apply_noise)) * 15).astype(np.int64)

        for i, pair in enumerate(self.dimerization):
            op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
            state = op(state)

            if apply_noise[i]:
                print('applied noise on pair', pair, 'gate', which_noise[i])
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.noise_operators[which_noise[i]], [pair])])
                state = op(state)

        assert np.isclose(np.dot(state.conj(), state), 1.0)

        return state


    def _initial_state_noisy_batched(self, n_states, fixed=True):
        states = np.zeros((n_states, 2 ** self.n_qubits), dtype=np.complex128)
        states[:, 0] = 1.

        if not fixed:        
            apply_noise = np.random.uniform(0, 1, size=(n_states, len(self.dimerization))) < self.config.noise_p
            which_noise = np.floor(np.random.uniform(0, 1, size=(n_states, len(self.dimerization))) * 15).astype(np.int64)

            for i, pair in enumerate(self.dimerization):
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                states = op(states.T).T

                for idx in range(n_states):
                    if apply_noise[idx, i]:
                        print('dimer: applied noise on pair', pair, 'gate', which_noise[idx, i])
                        op = ls.Operator(self.basis_bare, [ls.Interaction(self.noise_operators[which_noise[idx, i]], [pair])])
                        states[idx] = op(states[idx])
            return states



        for i, pair in enumerate(self.dimerization):
            op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
            states = op(states.T).T

            if self.apply_noise_dimer[i]:
                print('dimer: applied noise on pair', pair, 'gate', self.which_noise_dimer[i])
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.noise_operators[self.which_noise_dimer[i]], [pair])])
                states = op(states.T).T


        for idx in range(n_states):
            assert np.isclose(np.dot(states[idx].conj(), states[idx]), 1.0)

        return states




    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.
        pairs = []

        for n_layers in range(1):
            for shid, shift in enumerate([(0, 0), (1, 1), (1, 0), (0, 1)]):
                for pair in [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15)] if shid < 2 else [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
                #for pair in [(0, 4), (1, 5), (2, 6), (3, 7)] if shid < 2 else [(0, 1), (2, 3), (4, 5), (6, 7)]:
                    i, j = pair
                    xi, yi = i % self.Lx, i // self.Lx
                    xj, yj = j % self.Lx, j // self.Lx

                    xi = (xi + shift[0]) % self.Lx
                    xj = (xj + shift[0]) % self.Lx
                    yi = (yi + shift[1]) % self.Ly
                    yj = (yj + shift[1]) % self.Ly
                    ii, jj = xi + yi * self.Lx, xj + yj * self.Lx

                    layer = [((ii, jj), P_ij if self.unitary[ii, jj] == +1 else P_ijun)]
                    layers.append(deepcopy(layer))
                    pairs.append((ii, jj))

                
                for pair in [(0, 3), (1, 14), (2, 13), (3, 12), (4, 9), (5, 8), (6, 11), (7, 10)]:
                    i, j = pair
                    xi, yi = i % self.Lx, i // self.Ly
                    xj, yj = j % self.Lx, j // self.Ly

                    xi = (xi + shift[0]) % self.Lx
                    xj = (xj + shift[0]) % self.Lx
                    yi = (yi + shift[1]) % self.Ly
                    yj = (yj + shift[1]) % self.Ly
                    ii, jj = xi + yi * self.Ly, xj + yj * self.Ly

                    layer = [((ii, jj), P_ij if self.unitary[ii, jj] == +1 else P_ijun)]
                    layers.append(deepcopy(layer))
                    pairs.append((ii, jj))
                
        return layers, pairs

            

    def _initialize_parameters(self):
        if self.config.mode == 'fresh':
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.1 + np.pi / 2.

        if self.config.mode == 'preassigned':
            return self.config.start_params

        try:
            parameters_log = open(os.path.join(self.config.path_to_logs, 'parameters_log.dat'), 'r') 
            last_line = parameters_log.readlines()[-1]
            arr = 'np.array([' + last_line + '])'

            lambda_log = open(os.path.join(self.config.path_to_logs, 'lambda_log.dat'), 'r')
            last_line = lambda_log.readlines()[-1]
            self.lamb = float(last_line)
            return eval(arr)
        except:
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.1


    def _refresh_unitaries_derivatives(self, reduced = False):
        self.unitaries = []
        self.unitaries_herm = []
        self.derivatives = []

        for i in range(len(self.params)):
            unitaries_layer = []
            unitaries_herm_layer = []
            derivatives_layer = []

            if True: #i % 2 == 0:
                par = self.params[i]

                for pair, operator in self.layers[i]:
                    unitaries_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(1.0j * par * operator), [pair])]))
                    if not reduced:
                        unitaries_herm_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(-1.0j * par * operator), [pair])]))
                        derivatives_layer.append(ls.Operator(self.basis, [ls.Interaction(1.0j * operator, [pair])]))

            
            self.unitaries.append(unitaries_layer)
            if not reduced:
                self.unitaries_herm.append(unitaries_herm_layer)
                self.derivatives.append(derivatives_layer)
        return

    def init_circuit_qiskit(self, ancilla=False):
        if ancilla:
            return qiskit.QuantumCircuit(self.n_qubits + 1, 1)#self.n_qubits + 1)  # for Hadamard test scheme: ancilla qubit is the last
        return qiskit.QuantumCircuit(self.n_qubits, 1)#self.n_qubits)  # for survival rate scheme: no ancilla qubit


    def act_permutation_qiskit(self, circ, pair_permutations, ancilla=False, ancilla_qubit_idx = None):
        '''
            acts product of pair SWAP gates on the state
        '''

        if not ancilla:
            for pair in pair_permutations:
                circ.swap(pair[0], pair[1])

            return circ

        for pair in pair_permutations:
            circ.cswap(ancilla_qubit_idx, pair[0], pair[1])

        return circ


    def act_dimerization_qiskit(self, circ, inverse=False):
        '''
            acts D on the bare state |0> using the provided patterm
        '''

        assert self.spin == 0  # only the zero-spin-version so far
        if inverse:
            for pair in self.dimerization:
                i, j = pair

                circ.cx(i, j)
                circ.h(i)
                circ.x(j)
                circ.x(i)

            return circ

        for pair in self.dimerization:
            i, j = pair

            circ.x(i)
            circ.x(j)

            circ.h(i)
            circ.cx(i, j)
    

        return circ


    def act_psi_qiskit(self, circ, parameters, inverse=False):
        '''
            qubits has been initialized elsewhere;
            acts U(parameters) on the state using SWAPe gates
        '''

        for p, l in zip(reversed(parameters), reversed(self.layers)) if inverse else zip(parameters, self.layers):
            angle = -p if not inverse else p
            pair = l[0][0]

            i, j = pair


            eswap_op = qiskit.quantum_info.Operator([[np.exp(-1.0j * angle), 0, 0, 0],
                                                     [0, np.cos(angle), -1.0j * np.sin(angle), 0],
                                                     [0, -1.0j * np.sin(angle), np.cos(angle), 0],
                                                     [0, 0, 0, np.exp(-1.0j * angle)]])

            circ.unitary(eswap_op, [i, j], label='eswap')

        return circ

    def act_psi_qiskit_string(self, circ, parameters, ini, fin):
        '''
            qubits has been initialized elsewhere;
            acts U(parameters) on the state using SWAPe gates
        '''

        ctr = 0
        for p, l in zip(parameters, self.layers):
            if ctr < ini or ctr >= fin:
                ctr += 1
                continue
            angle = p
            pair = l[0][0]

            i, j = pair


            eswap_op = qiskit.quantum_info.Operator([[np.exp(-1.0j * angle), 0, 0, 0],
                                                     [0, np.cos(angle), -1.0j * np.sin(angle), 0],
                                                     [0, -1.0j * np.sin(angle), np.cos(angle), 0],
                                                     [0, 0, 0, np.exp(-1.0j * angle)]])

            circ.unitary(eswap_op, [i, j], label='eswap')
            ctr += 1

        return circ


    def act_derivative_qiskit(self, circ, site, ancilla_qubit_idx):
        i, j = self.layers[site][0][0]

        ifredkin = np.zeros((8, 8), dtype=np.complex128)
        ifredkin[0, 0] = 1.
        ifredkin[1, 1] = 1.0j
        ifredkin[2, 2] = 1.
        ifredkin[3, 5] = ifredkin[5, 3] = 1.0j
        ifredkin[4, 4] = 1.
        ifredkin[6, 6] = 1.
        ifredkin[7, 7] = 1.0j

        ifredkin_op = qiskit.quantum_info.Operator(ifredkin)

        circ.unitary(ifredkin_op, [ancilla_qubit_idx, i, j], label='ifredkin')

        return circ





class SU2_symmetrized_hexagon(SU2_symmetrized):
    def __init__(self, subl, Lx, Ly, basis, config, unitary, BC, spin=0):
        super().__init__(1, 6, 1, basis, config, unitary, BC, spin)
        self.n_qubits = 6

        return


    def _initial_state(self):
        if self.ini_state is not None:
            return self.ini_state.copy()

        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[0] = 1.

        if self.spin == 0:
            for pair in self.dimerization:
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                state = op(state)
        elif self.spin == 1:
            for idx, pair in enumerate(self.dimerization):
                if idx != 0:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                else:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(self.tripletizer, [pair])])
                state = op(state)
        else:
            for pair in [(0, 1)]:
                op = ls.Operator(self.basis_bare, [ls.Interaction(self.singletizer, [pair])])
                state = op(state)
            op = ls.Operator(self.basis_bare, [ls.Interaction(self.octupletizer, [(2, 3, 4, 5)])])
            state = op(state)

        assert np.isclose(np.dot(state.conj(), state), 1.0)

        ### perform the unitary transform ###
        for site, phase in enumerate(self.unitary_site):
            op = ls.Operator(self.basis_bare, [ls.Interaction(sz, [(site,)])])
            if phase == -1:
                state = op(state)

        state_su2 = np.zeros(self.basis.number_states, dtype=np.complex128)
        for i in range(self.basis.number_states):
            x = self.basis.states[i]
            _, _, norm = self.basis.state_info(x)
            state_su2[i] = state[self.basis_bare.index(x)] / norm

        assert np.isclose(np.dot(state_su2.conj(), state_su2), 1.0)
        assert np.isclose(np.dot(state_su2.conj(), self.total_spin(state_su2)) + 3. * self.Lx * self.Ly, self.spin * (self.spin + 1) * 4.)

        return state_su2


    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.

        for pattern in [[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (0, 5)], [(1, 3), (0, 4)], [(2, 4), (3, 5)], [(0, 4), (1, 3)]]:                
            for pair in pattern:
                i, j = pair

                layer = [((i, j), P_ij if self.unitary[i, j] == +1 else P_ijun)]
                layers.append(deepcopy(layer))
        return layers


class SU2_symmetrized_honeycomb_2x2(SU2_symmetrized):
    def __init__(self, subl, Lx, Ly, basis, config, unitary, BC, spin=0):
        super().__init__(subl, Lx, Ly, basis, config, unitary, BC, spin)
        self.n_qubits = 2 * Lx* Ly

        return

    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.

        for pattern in [
                        [(0, 7), (2, 5), (3, 4), (1, 6)], \
                        [(0, 2), (1, 3), (4, 6), (5, 7)], \
                        [(0, 5), (2, 7), (1, 4), (3, 6)], \
                        [(0, 4), (1, 5), (2, 6), (3, 7)], \
                        [(0, 1), (2, 3), (4, 5), (6, 7)], \
                        [(0, 6), (2, 4), (1, 7), (3, 5)]
                ]:
            for pair in pattern:
                i, j = pair

                layer = [((i, j), P_ij if self.unitary[i, j] == +1 else P_ijun)]
                layers.append(deepcopy(layer))
        return layers

class SU2_symmetrized_honeycomb_3x3(SU2_symmetrized):
    def __init__(self, subl, Lx, Ly, basis, config, unitary, BC, spin=0):
        super().__init__(subl, Lx, Ly, basis, config, unitary, BC, spin)
        self.n_qubits = 2 * Lx* Ly

        return

    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.


        bonds = [(0, 1), (0, 13), (0, 15), (1, 6), (1, 10), (2, 3), (2, 15), (2, 17), (3, 6), (3, 8), (4, 5), (4, 13), (4, 17), (5, 8), (5, 10), (6, 7), (7, 12), (7, 16), (8, 9), (9, 12), (9, 14), (10, 11), (11, 14), (11, 16), (12, 13), (14, 15), (16, 17)]
        #bonds_j2 = [(0, 2), (0, 6), (0, 14), (0, 12), (0, 4), (0, 10), (1, 3), (1, 15), (1, 13), (1, 5), (1, 7), (1, 11), (2, 14), (2, 16), (2, 4), (2, 8), (2, 6), (3, 15), (3, 17), (3, 5), (3, 9), (3, 7), (4, 8), (4, 10), (4, 12), (4, 16), (5, 17), (5, 13), (5, 11), (5, 9), (6, 8), (6, 12), (6, 16), (6, 10), (7, 9), (7, 13), (7, 17), (7, 11), (8, 10), (8, 14), (8, 12), (9, 10), (9, 11), (9, 15), (9, 13), (10, 14), (10, 16), (11, 15), (11, 17), (12, 14), (12, 16), (13, 15), (13, 17), (14, 16)]
        for pattern in [
                    [(0, 15), (2, 17), (4, 13), (6, 3), (8, 5), (10, 1), (12, 9), (14, 11), (16, 7)], \
                    [(0, 13), (2, 15), (4, 17), (6, 1), (8, 3), (10, 5), (12, 7), (14, 9), (16, 11)], \
                    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)], \
                    [(0, 15), (2, 17), (4, 13), (6, 3), (8, 5), (10, 1), (12, 9), (14, 11), (16, 7)], \
                    [(0, 13), (2, 15), (4, 17), (6, 1), (8, 3), (10, 5), (12, 7), (14, 9), (16, 11)], \
                    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)], \
                ]:
            for pair in pattern:
                i, j = pair

                layer = [((i, j), P_ij if self.unitary[i, j] == +1 else P_ijun)]
                layers.append(deepcopy(layer))
        return layers

