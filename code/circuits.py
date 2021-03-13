import numpy as np
from copy import deepcopy
import scipy as sp
from scipy import sparse
import lattice_symmetries as ls
import scipy
import utils

sz = np.array([[1, 0], \
               [0, -1]])

sx = np.array(
               [[0, 1], \
               [1, 0]]
              )

sy = np.array([[0, 1.0j], [-1.0j, 0]])

s0 = np.eye(2)

SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)

class Circuit(object):
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=n_qubits, hamming_weight=None)
        self.basis.build()

    def __call__(self):
        state = self._initial_state()
        for gate in self.unitaries:
            state = gate(state)
        assert np.isclose(np.dot(state, self.total_spin(state)) + 3 * self.Lx * self.Ly, 0.0)
        assert np.isclose(np.dot(state, state), 1.0)
        return state

    def get_all_derivatives(self, hamiltonian):
        LEFT = hamiltonian(self.__call__())
        for u_herm in reversed(self.unitaries_herm):
            LEFT = u_herm(LEFT)
        LEFT = LEFT.conj()

        RIGHT = self._initial_state()

        grads = []
        for i in range(len(self.derivatives)):
            grads.append(np.dot(LEFT, self.derivatives[i](RIGHT)))

            RIGHT = self.unitaries[i](RIGHT)
            LEFT = (self.unitaries[i](LEFT.conj())).conj()

        assert np.allclose(np.array(grads).imag, np.array(grads) * 0.0)

        return 2 * np.array(grads).real

    def get_natural_gradients(self, hamiltonian):
        grads = self.get_all_derivatives(hamiltonian)
        ij, j = self.get_metric_tensor()
        G = ij - np.einsum('i,j->ij', j.conj(), j)
        return grads, ij, j

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


class SU2_PBC_symmetrized(Circuit):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        super().__init__(Lx * Ly)
        self.tr_x = utils.get_x_symmetry_map(self.Lx, self.Ly)
        self.tr_y = utils.get_y_symmetry_map(self.Lx, self.Ly)

        # init initial state of the circuit
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

        return state


    def get_all_derivatives(self, hamiltonian):
        LEFT = hamiltonian(self.__call__())
        for layer in reversed(self.unitaries_herm):
            for u_herm in reversed(layer):
                LEFT = u_herm(LEFT)
        LEFT = LEFT.conj()

        RIGHT = self._initial_state()


        grads = []
        for idx, layer in enumerate(self.layers):
            derivative = RIGHT * 0.0
            for der in self.derivatives[idx]:
                derivative += der(RIGHT)
            grads.append(np.dot(LEFT, derivative))


            LEFT = LEFT.conj()
            for u in self.unitaries[idx]:
                RIGHT = u(RIGHT)
                LEFT = u(LEFT)
            LEFT = LEFT.conj()

        return 2 * np.array(grads).real


    def get_metric_tensor(self):
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

            for layer in reversed(self.unitaries_herm):
                for u_herm in reversed(layer):
                    LEFT = u_herm(LEFT) # LEFT = L^+_0 L^+_1 ... L^+_{N - 1} L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>

            RIGHT = self._initial_state()  # RIGHT = |0>
            for j in range(len(self.params)):
                derivative = RIGHT * 0.
                for der in self.derivatives[j]:
                    derivative += der(RIGHT)

                MT[i, j] = np.dot(LEFT.conj(), derivative)

                for u in self.unitaries[j]:
                    RIGHT = u(RIGHT) # LEFT = L^+_{j + 1} ... L^+_{N - 1} L_{N - 1} .. L_i A_i L_{i-1} ... L_0 |0>
                    LEFT = u(LEFT) # RIGHT = L_j ... L_0 |0>


        der_i = np.zeros(len(self.params), dtype=np.complex128)
        LEFT = self.__call__() # L_{N-1} ... L_0 |0>

        for layer in reversed(self.unitaries_herm):
            for u_herm in reversed(layer):
                LEFT = u_herm(LEFT) # LEFT = L^+_0 L^+_1 ... L^+_{N - 1} L_{N - 1} .. L_i L_{i-1} ... L_0 |0>

        RIGHT = self._initial_state()   # RIGHT = |0>
        for i in range(len(self.params)):
            derivative = RIGHT * 0. + 0.0j
            for der in self.derivatives[i]:
                derivative += der(RIGHT)

            der_i[i] = np.dot(LEFT.conj(), derivative)

            for u in self.unitaries[i]:
                RIGHT = u(RIGHT) # LEFT = L^+_{i + 1} ... L^+_{N - 1} L_{N - 1} ... L_0 |0>
                LEFT = u(LEFT) # RIGHT = L_i ... L_0 |0>

        # MT -= np.einsum('i,j->ij', der_i.conj(), der_i)

        return MT, der_i

    def _initial_state(self):
        if self.ini_state is not None:
            return self.ini_state.copy()

        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        state[0] = 1.

        singletizer = np.zeros((4, 4), dtype=np.complex128)
        singletizer[1, 0] = 1. / np.sqrt(2)
        singletizer[2, 0] = -1. / np.sqrt(2)
        singletizer = singletizer + singletizer.T

        for i in np.arange(self.Lx * self.Ly)[::2]:
            op = ls.Operator(self.basis, [ls.Interaction(singletizer, [(i, i + 1)])])
            state = op(state)

        assert np.isclose(np.dot(state.conj(), state), 1.0)

        all_bonds = []
        for i in range(self.Lx * self.Ly):
            for j in range(self.Lx * self.Ly):
                if i == j:
                    continue
                all_bonds.append((i, j))

        self.total_spin = ls.Operator(self.basis, [ls.Interaction(SS, all_bonds)])
        assert np.isclose(np.dot(state.conj(), self.total_spin(state)) + 3 * self.Lx * self.Ly, 0.0)
        assert np.allclose(state, state[self.tr_x[self.tr_x]])
        assert np.allclose(state, state[self.tr_y[self.tr_y]])

        return state

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

                distances[i, j] = distance  # PBC

        assert np.allclose(distances, distances.T)

        return np.around(distances, decimals=4)

    def _get_dimerizarion_layers(self):
        layers = []


        ### up ###
        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 1:
                continue
            i_01 = (x + 0) % self.Lx + ((y + 1) % self.Lx) * self.Lx
            layer.append((i, i_01))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 0:
                continue
            i_01 = (x + 0) % self.Lx + ((y + 1) % self.Lx) * self.Lx
            layer.append((i, i_01))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        ### right ###
        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if x % 2 == 1:
                continue
            i_10 = (x + 1) % self.Lx + ((y + 0) % self.Lx) * self.Lx
            layer.append((i, i_10))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if x % 2 == 0:
                continue
            i_10 = (x + 1) % self.Lx + ((y + 0) % self.Lx) * self.Lx
            layer.append((i, i_10))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        ### diagonal_upup ###
        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 0:
                continue
            i_11 = (x + 1) % self.Lx + ((y + 1) % self.Lx) * self.Lx
            layer.append((i, i_11))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 1:
                continue
            i_11 = (x + 1) % self.Lx + ((y + 1) % self.Lx) * self.Lx
            layer.append((i, i_11))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly


        ### diagonal_updown ###
        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 0:
                continue
            i_11 = (x + 1) % self.Lx + ((y - 1) % self.Lx) * self.Lx
            layer.append((i, i_11))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        layer = []
        for i in range(self.Lx * self.Ly):
            x, y = i % self.Lx, i // self.Ly

            if y % 2 == 1:
                continue
            i_11 = (x + 1) % self.Lx + ((y - 1) % self.Lx) * self.Lx
            layer.append((i, i_11))
        layers.append(deepcopy(layer))
        b = np.array([i for sub in layer for i in sub])
        assert len(np.unique(b)) == self.Lx * self.Ly



        return layers

    def _initialize_parameters(self):
        return np.random.uniform(size=len(self.layers))

    def _refresh_unitaries_derivatives(self):
        self.unitaries = []
        self.unitaries_herm = []
        self.derivatives = []

        P_ij = (SS + np.eye(4)) / 2.
        for layer, par in zip(self.layers, self.params):
            unitaries_layer = []
            unitaries_herm_layer = []
            derivatives_layer = []

            for pair in layer:
                unitaries_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(1.0j * par * P_ij), [pair])]))
                unitaries_herm_layer.append(ls.Operator(self.basis, [ls.Interaction(scipy.linalg.expm(-1.0j * par * P_ij), [pair])]))
                derivatives_layer.append(ls.Operator(self.basis, [ls.Interaction(1.0j * P_ij, [pair])]))
            self.unitaries.append(unitaries_layer)
            self.unitaries_herm.append(unitaries_herm_layer)
            self.derivatives.append(derivatives_layer)

        return 