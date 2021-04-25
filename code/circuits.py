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

    def get_natural_gradients(self, hamiltonian, projector, N_samples=None):
        t = time()
        ij, j, ij_sampling, j_sampling = self.get_metric_tensor(projector, N_samples)
        print('metric tensor: ', time() - t)
        self.connectivity_sampling = j_sampling

        t = time()
        grads, grads_sampling = self.get_all_derivatives(hamiltonian, projector, N_samples)
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
    def __init__(self, subl, Lx, Ly, basis, config, unitary, BC, spin=0):
        self.BC = BC
        self.Lx = Lx
        self.Ly = Ly
        self.n_subl = subl
        self.n_qubits = Lx * Ly * subl
        self.spin = spin
        self.dimerization = config.dimerization
        super().__init__(Lx * Ly * subl, basis, config, unitary)
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
        self.layers = self._get_dimerizarion_layers()
        self.params = self._initialize_parameters()


        ### defining of unitaries ###
        self._refresh_unitaries_derivatives()

        return

    def __call__(self, from_idx = None):
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
        t = time()
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

        
        if self.config.test:
            return MT, der_i, MT_sample / norm_sampling, connectivity / norm_sampling
        return None, None, MT_sample / norm_sampling, connectivity / norm_sampling

    def get_metric_tensor_sampling(self, projector, N_samples):
        ### get states ###
        self.der_states = []
        new_params = self.params.copy()

        t = time()
        for i in range(len(self.params)):
            new_params[i] += np.pi / 2.
            self.set_parameters(new_params)
            self.der_states.append(self.__call__(from_idx=i))
            new_params[i] -= np.pi / 2.
        self.set_parameters(new_params)


        print('obtain states for MT ', time() - t)
        metric_tensor = utils.compute_metric_tensor_sample(self.der_states, projector, N_samples)
        metric_tensor = metric_tensor + metric_tensor.conj().T
        metric_tensor -= np.diag(np.diag(metric_tensor)) / 2.

        connectivity = utils.compute_connectivity_sample(self.__call__(), self.der_states, projector, N_samples, theta=0.) + \
                        1.0j * utils.compute_connectivity_sample(self.__call__(), self.der_states, projector, N_samples, theta=-1)
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

    def _get_dimerizarion_layers(self):
        layers = []
        P_ij = (SS + np.eye(4)) / 2.
        P_ijun = (SSun + np.eye(4)) / 2.


        for n_layers in range(1):
            
            for shid, shift in enumerate([(0, 0), (1, 1), (1, 0), (0, 1)]):
                for pair in [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15)] if shid < 2 else [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
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
            return layers
            

    def _initialize_parameters(self):
        if self.config.mode == 'fresh':
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.1

        if self.config.mode == 'preassigned':
            return self.config.start_params

        try:
            parameters_log = open(os.path.join(self.config.path_to_logs, 'parameters_log.dat'), 'r') 
            last_line = parameters_log.readlines()[-1]
            arr = 'np.array([' + last_line + '])'
            return eval(arr)
        except:
            return (np.random.uniform(size=len(self.layers)) - 0.5) * 0.1


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

            
            self.unitaries.append(unitaries_layer)
            self.unitaries_herm.append(unitaries_herm_layer)
            self.derivatives.append(derivatives_layer)
        return


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

