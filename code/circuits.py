import numpy as np
from copy import deepcopy
import os
import scipy as sp
from scipy import sparse
import lattice_symmetries as ls
import scipy
import utils
from time import time

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

class Circuit(object):
    def __init__(self, n_qubits, basis, config, unitary, **kwargs):
        self.basis = basis
        self.config = config
        self.n_qubits = n_qubits
        self.unitary = unitary
        self.unitary_site = unitary[0, :]

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

    def get_natural_gradients(self, hamiltonian, projector, N_samples=None):
        ij, j, ij_sampling, j_sampling = self.get_metric_tensor(projector, N_samples)
        self.connectivity_sampling = j_sampling
        grads, grads_sampling = self.get_all_derivatives(hamiltonian, projector, N_samples)

        if N_samples is None:
            return grads, ij, j
        return grads, ij, j, grads_sampling, ij_sampling, j_sampling

    def _get_derivative_idx(self, param_idx):
        return self.derivatives[param_idx]

    def _initialize_parameters(self):
        return np.random.uniform(size=len(self.locs))

    def get_parameters(self):
        return self._pack_parameters()

    def set_parameters(self, parameters):
        self.params = parameters.copy()
        self._refresh_unitaries_derivatives()
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
    def __init__(self, Lx, Ly, basis, config, unitary, BC, spin=0):
        self.BC = BC
        self.Lx = Lx
        self.Ly = Ly
        self.n_qubits = Lx * Ly
        self.spin = spin
        super().__init__(Lx * Ly, basis, config, unitary)
        #self.tr_x = utils.get_x_symmetry_map(self.Lx, self.Ly)
        #self.tr_y = utils.get_y_symmetry_map(self.Lx, self.Ly)
        #self.Cx = utils.get_Cx_symmetry_map(self.Lx, self.Ly)
        #self.Cy = utils.get_Cy_symmetry_map(self.Lx, self.Ly)
        
        

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


        self.ini_state = None
        self.ini_state = self._initial_state()


        self.pairwise_distances = self.get_pairwise_distances()

        ### defining operator locs ###
        self.layers = self._get_dimerizarion_layers()
        self.params = self._initialize_parameters()


        ### defining of unitaries ###
        self._refresh_unitaries_derivatives()

        return

    def __call__(self):
        state = self._initial_state()
        # assert np.isclose(np.dot(state.conj(), state), 1.0)
        for layer in self.unitaries:
            for gate in layer:
                state = gate(state)
                # assert np.isclose(np.dot(state.conj(), state), 1.0)

        #assert np.allclose(state, state[self.tr_x])
        #assert np.allclose(state, state[self.tr_y])
        #assert np.allclose(state, state[self.tr_x[self.tr_x]])
        #assert np.allclose(state, state[self.tr_y[self.tr_y]])
        #assert np.allclose(state, state[self.Cx])
        #assert np.allclose(state, state[self.Cy])
        return state


    def get_all_derivatives(self, hamiltonian, projector, N_samples):
        state = self.__call__()
        state_proj = projector(state)
        norm = np.dot(state.conj(), state_proj)
        print('norm = ', norm)

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


        if N_samples is None:
            return np.array(grads), None

        derivatives_sampling = utils.compute_energy_der_sample(self.__call__(), self.der_states, hamiltonian, projector, N_samples)
        norm_sampling = utils.compute_norm_sample(self.__call__(), projector, N_samples)
        energy_sampling = utils.compute_energy_sample(self.__call__(), hamiltonian, projector, N_samples)

        grad_sampling = (derivatives_sampling / norm_sampling - self.connectivity_sampling * energy_sampling / norm_sampling).real * 2.



        #for i in range(len(self.params)):
        #    print((derivatives_sampling[i] / norm_sampling).real * 2., (numerators[i] / norm).real * 2., '|', \
        #           (self.connectivity_sampling[i] * energy_sampling / norm_sampling).real * 2., (numerators_conn[i] / norm * energy).real * 2.)
        #    print(grad_sampling[i], np.array(grads)[i])
        #print('energy sampling:', energy_sampling / norm_sampling - 33, 'energy exact', energy - 33)
        #exit(-1)


        ### sampling testing ###
        '''
        new_params = self.params.copy()
        for i in range(len(self.params)):
            new_params[i] += np.pi / 4.
            self.set_parameters(new_params)
            state = self.__call__()
            state_proj = projector(state)

            energy_plus = np.dot(np.conj(state), hamiltonian(state_proj))
            norm_plus = np.dot(np.conj(state), state_proj)
            
            energy_plus_sample = utils.compute_energy_sample(self.__call__(), hamiltonian, projector, N_samples)
            norm_plus_sample = utils.compute_norm_sample(self.__call__(), projector, N_samples)

            new_params[i] -= np.pi / 2.
            self.set_parameters(new_params)
            energy_minus_sample = utils.compute_energy_sample(self.__call__(), hamiltonian, projector, N_samples)
            norm_minus_sample = utils.compute_norm_sample(self.__call__(), projector, N_samples)
            state = self.__call__()
            state_proj = projector(state)

            energy_minus = np.dot(np.conj(state), hamiltonian(state_proj))
            norm_minus = np.dot(np.conj(state), state_proj)

            new_params[i] += np.pi / 4.
            self.set_parameters(new_params)

            print('energy derivative', i, numerators[i].real * 2., energy_plus - energy_minus, energy_plus_sample - energy_minus_sample)
            print('connectivity', i, numerators_conn[i].real * 2, norm_plus - norm_minus, norm_plus_sample - norm_minus_sample)

        '''
        return np.array(grads), grad_sampling


    def get_metric_tensor(self, projector, N_samples):
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

        if N_samples is None:
            return MT, der_i, None, None

        # MT -= np.einsum('i,j->ij', der_i.conj(), der_i)

        MT_sample, connectivity = self.get_metric_tensor_sampling(projector, N_samples)
        norm_sampling = utils.compute_norm_sample(self.__call__(), projector, N_samples)
        #print(np.save('test_MT.npy', MT_sample))
        #assert np.allclose(MT_sample, MT_sample.conj().T)
        #for i in range(MT.shape[0]):
        #    for j in range(MT.shape[1]):
        #        print(MT[i, j] * norm, MT_sample[i, j])
        
        #for i in range(connectivity.shape[0]):
        #    print(der_i[i] * norm, connectivity[i])

        

        return MT, der_i, MT_sample / norm_sampling, connectivity / norm_sampling

    def get_metric_tensor_sampling(self, projector, N_samples):
        ### get states ###
        self.der_states = []
        new_params = self.params.copy()

        t = time()
        for i in range(len(self.params)):
            new_params[i] += np.pi / 2.
            self.set_parameters(new_params)
            self.der_states.append(self.__call__())
            new_params[i] -= np.pi / 2.
        self.set_parameters(new_params)
        print('obtain states for MT ', time() - t)
        metric_tensor = utils.compute_metric_tensor_sample(self.der_states, projector, N_samples, theta=0.) + \
                        1.0j * utils.compute_metric_tensor_sample(self.der_states, projector, N_samples, theta=-np.pi / 2.)
        connectivity = utils.compute_connectivity_sample(self.__call__(), self.der_states, projector, N_samples, theta=0.) + \
                        1.0j * utils.compute_connectivity_sample(self.__call__(), self.der_states, projector, N_samples, theta=-np.pi / 2.)
        return metric_tensor, connectivity


    def _initial_state(self):
        if self.ini_state is not None:
            return self.ini_state.copy()

        '''
        state1 = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state2 = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        
        spin_1 = np.zeros(self.n_qubits, dtype=np.int64)
        spin_2 = np.zeros(self.n_qubits, dtype=np.int64)
        spin_3 = np.zeros(self.n_qubits, dtype=np.int64)
        spin_4 = np.zeros(self.n_qubits, dtype=np.int64)
        for i in range(self.n_qubits):
            x, y = i % self.Lx, i // self.Lx
            if x % 2 == 0:
                spin_1[i] = 1.
            else:
                spin_2[i] = 1.

            if y % 2 == 0:
                spin_3[i] = 1.
            else:
                spin_4[i] = 1.

        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[utils.spin_to_index(spin_1, number_spins = self.n_qubits)] = 1. / 2
        state[utils.spin_to_index(spin_2, number_spins = self.n_qubits)] = 1. / 2#np.sqrt(2)
        state[utils.spin_to_index(spin_3, number_spins = self.n_qubits)] = 1. / 2
        state[utils.spin_to_index(spin_4, number_spins = self.n_qubits)] = 1. / 2

        state_su2 = np.zeros(self.basis.number_states, dtype=np.complex128)
        for i in range(self.basis.number_states):
            x = self.basis.states[i]
            _, _, norm = self.basis.state_info(x)
            state_su2[i] = state[self.basis_bare.index(x)] / norm
            #print(state[self.basis_bare.index(x)] / norm)

        #print(self.spin, np.dot(state_su2.conj(), state_su2), np.dot(state.conj(), state))
        assert np.isclose(np.dot(state_su2.conj(), state_su2), 1.0)

        

        return state_su2
        

        
        state1[0] = 1.
        state2[0] = 1.
        ''' 
        #hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        #for i in np.arange(self.Lx * self.Ly):
        #    op = ls.Operator(self.basis, [ls.Interaction(hadamard, [(i,)])])
        #    state = op(state)
        

        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[0] = 1.

        singletizer = np.zeros((4, 4), dtype=np.complex128)
        singletizer[1, 0] = 1. / np.sqrt(2)
        singletizer[2, 0] = -1. / np.sqrt(2)
        singletizer = singletizer + singletizer.T

        tripletizer = np.zeros((4, 4), dtype=np.complex128)
        #tripletizer[1, 0] = +1. / np.sqrt(2)
        #tripletizer[2, 0] = +1. / np.sqrt(2)
        tripletizer[3, 0] = 1.
        tripletizer = tripletizer + tripletizer.T

        octupletizer = np.zeros((16, 16), dtype=np.complex128)
        #octupletizer[12, 0] = +1. / np.sqrt(6)
        #octupletizer[9, 0] = +1. / np.sqrt(6)
        #octupletizer[3, 0] = +1. / np.sqrt(6)
        #octupletizer[6, 0] = +1. / np.sqrt(6)
        #octupletizer[10, 0] = +1. / np.sqrt(6)
        #octupletizer[5, 0] = +1. / np.sqrt(6)
        octupletizer[15, 0] = 1.

        octupletizer = octupletizer + octupletizer.T

        
        if self.spin == 0:
            #for i in np.arange(self.Lx * self.Ly)[::2]:
            for pair in [(0, 5), (1, 4), (2, 7), (3, 6), (8, 13), (9, 12), (10, 15), (11, 14)]:
            #for pair in [(5, 10), (9, 6), (1, 4), (2, 7), (8, 13), (11, 14), (0, 15), (3, 12)]:
            #for pair in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
                op = ls.Operator(self.basis_bare, [ls.Interaction(singletizer, [pair])])
                state = op(state)
        elif self.spin == 1:
            for i in np.arange(self.Lx * self.Ly)[::2]:
                if i != 4:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(singletizer, [(i, i + 1)])])
                else:
                    op = ls.Operator(self.basis_bare, [ls.Interaction(tripletizer, [(i, i + 1)])])
                state = op(state)
        else:
            for pair in [(0, 1), (2, 3), (4, 8), (7, 11), (12, 13), (14, 15)]:
                op = ls.Operator(self.basis_bare, [ls.Interaction(singletizer, [pair])])
                state = op(state)
            op = ls.Operator(self.basis_bare, [ls.Interaction(octupletizer, [(5, 6, 9, 10)])])
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
            #print(state[self.basis_bare.index(x)] / norm)

        #print(self.spin, np.dot(state_su2.conj(), state_su2), np.dot(state.conj(), state))
        assert np.isclose(np.dot(state_su2.conj(), state_su2), 1.0)

        if self.spin == 0.0:
            assert np.isclose(np.dot(state_su2.conj(), self.total_spin(state_su2)) + 3. * self.Lx * self.Ly, 0.0)
        elif self.spin == 1.0:
            assert np.isclose(np.dot(state_su2.conj(), self.total_spin(state_su2)) + 3. * self.Lx * self.Ly, 2.0 * 4.)
        else:
            assert np.isclose(np.dot(state_su2.conj(), self.total_spin(state_su2)) + 3. * self.Lx * self.Ly, 6.0 * 4.)


        
        return state_su2

    def get_pairwise_distances(self):
        distances = np.zeros((self.Lx * self.Ly, self.Lx * self.Ly))

        for i in range(self.Lx * self.Ly):
            for j in range(self.Lx * self.Ly):
                xi, yi = i % self.Lx, i // self.Lx
                xj, yj = j % self.Lx, j // self.Lx

                
                distance = np.min([\
                                   np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), \
                                   np.sqrt((xi - xj + self.Lx) ** 2 + (yi - yj) ** 2), \
                                   np.sqrt((xi - xj - self.Lx) ** 2 + (yi - yj) ** 2), \
                                   np.sqrt((xi - xj) ** 2 + (yi - yj + self.Ly) ** 2), \
                                   np.sqrt((xi - xj) ** 2 + (yi - yj - self.Ly) ** 2), \
                                   np.sqrt((xi - xj - self.Lx) ** 2 + (yi - yj - self.Ly) ** 2), \
                                   np.sqrt((xi - xj - self.Lx) ** 2 + (yi - yj + self.Ly) ** 2), \
                                   np.sqrt((xi - xj + self.Lx) ** 2 + (yi - yj + self.Ly) ** 2), \
                                   np.sqrt((xi - xj + self.Lx) ** 2 + (yi - yj - self.Ly) ** 2)
                                  ])
                distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)  # OBC

                distances[i, j] = distance  # PBC

        assert np.allclose(distances, distances.T)

        return np.around(distances, decimals=4)


    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.


        for n_layers in range(1):
            
            for shift in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                for pair in [(0, 12), (1, 5), (2, 6), (3, 15), (4, 7), (8, 11), (9, 13), (10, 14)]:
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

                for pair in [(0, 15), (1, 14), (2, 13), (3, 12), (4, 9), (5, 8), (6, 11), (7, 10)]:
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

        '''
            for shift in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                for pair in [(0, 12), (1, 5), (2, 6), (3, 15), (4, 7), (8, 11), (9, 13), (10, 14)]:
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
        '''
        return layers
        '''
            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 0:
                    continue
                i_to = (x + 1) % self.Lx + ((y + 1) % self.Lx) * self.Lx
                if (y + 1 < self.Ly and x + 1 < self.Lx) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 0:
                    continue
                i_to = (x + 1) % self.Lx + ((y - 1) % self.Lx) * self.Lx
                if (x + 1 < self.Lx and y > 0) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 1:
                    continue
                i_to = (x + 1) % self.Lx + ((y + 1) % self.Lx) * self.Lx
                if (y + 1 < self.Ly and x + 1 < self.Lx) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 1:
                    continue
                i_to = (x + 1) % self.Lx + ((y - 1) % self.Lx) * self.Lx
                if (x + 1 < self.Lx and y > 0) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))
        

            
            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 0:
                    continue
                i_to = (x + 0) % self.Lx + ((y + 1) % self.Lx) * self.Lx
                if (y + 1 < self.Ly) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if y % 2 == 1:
                    continue
                i_to = (x + 0) % self.Lx + ((y + 1) % self.Lx) * self.Lx
                if (y + 1 < self.Ly) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if x % 2 == 0:
                    continue
                i_to = (x + 1) % self.Lx + ((y + 0) % self.Lx) * self.Lx
                if (x + 1 < self.Lx) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

            for i in range(self.Lx * self.Ly):
                layer = []
                x, y = i % self.Lx, i // self.Ly

                if x % 2 == 1:
                    continue
                i_to = (x + 1) % self.Lx + ((y + 0) % self.Lx) * self.Lx
                if (x + 1 < self.Lx) or self.BC == 'PBC':
                    layer.append(((i, i_to), P_ij if self.unitary[i, i_to] == +1 else P_ijun))
                    layers.append(deepcopy(layer))

        return layers
        '''



    def _initialize_parameters(self):
        if self.config.mode == 'random':
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.03
        try:
            parameters_log = open(os.path.join(self.config.path_to_logs, 'parameters_log.dat'), 'r') 
            last_line = parameters_log.readlines()[-1]
            arr = 'np.array([' + last_line + '])'
            return eval(arr)
        except:
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.03


    def _refresh_unitaries_derivatives(self):
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
                    unitaries_herm_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(-1.0j * par * operator), [pair])]))
                    derivatives_layer.append(ls.Operator(self.basis, [ls.Interaction(1.0j * operator, [pair])]))

            
            #else:
            #    par = self.params[i]

            #    #### adding z-interlayers ####
            #    for i in range(self.Lx * self.Ly):
            #        unitaries_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(1.0j * par * sz), [(i,)])]))
            #        unitaries_herm_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(-1.0j * par * sz), [(i,)])]))
            #        derivatives_layer.append(ls.Operator(self.basis, [ls.Interaction(1.0j * sz, [(i,)])]))
            self.unitaries.append(unitaries_layer)
            self.unitaries_herm.append(unitaries_herm_layer)
            self.derivatives.append(derivatives_layer)

        return

