import numpy as np
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

class Circuit(object):
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=n_qubits, hamming_weight=None)
        self.basis.build()

    def __call__(self):
        state = self._initial_state()
        for gate in self.unitaries:
            state = gate(state)

        return state

    def derivative(self, param_idx):
        # print(param_idx, flush=True)
        dev = self._get_derivative_idx(param_idx)

        state = self._initial_state()
        for gate in self.unitaries[:param_idx]:
            state = gate(state)

        state = dev(state)

        for gate in self.unitaries[param_idx:]:
            state = gate(state)

        return state

    def get_all_derivatives(self, hamiltonian):
        LEFT = hamiltonian(self.__call__())
        for u_herm in reversed(self.unitaries_herm):
            LEFT = u_herm(LEFT)
        LEFT = LEFT.conj()

        RIGHT = self._initial_state()

        grads = []
        for i in range(len(self.derivatives)):
            # assert np.allclose(LEFT, self.unitaries[i](self.unitaries_herm[i](LEFT)))
            grads.append(np.dot(LEFT, self.derivatives[i](RIGHT)))

            RIGHT = self.unitaries[i](RIGHT)
            LEFT = (self.unitaries[i](LEFT.conj())).conj()

        return 2 * np.array(grads).real

    def _get_derivative_idx(self, param_idx):
        raise NotImplementedError()

    def _define_parameter_site_bonds(self, **kwargs):
        raise NotImplementedError()

    def _get_unitaries(self, **kwargs):
        raise NotImplementedError()

    def _initialize_parameters(self):
        raise NotImplementedError()

    def get_parameters(self):
        return self._pack_parameters()

    def set_parameters(self, parameters):
        raise NotImplementedError()

    def _unpack_parameters(self, params_vectorized):
        raise NotImplementedError()

    def _pack_parameters(self):
        raise NotImplementedError()

    def _get_all_derivatives(self):
        raise NotImplementedError()

    def _initial_state(self):
        raise NotImplementedError()

class TrotterizedMarshallsSquareHeisenbergNNAFM(Circuit):
    '''
        constructs unitary circuit
        U(\\theta_i, \\theta_perp_lm, \\theta_Z_lm) = \\prod_i e^{-i \\theta_i Z_i} \\prod_{l < m}  A_lm(\\theta_perp_lm) B_lm(\\thetaZ_lm)
        where B_lm(\\thetaZ_lm) = exp(-i Z_l Z_m \\theta_lm)
              A_lm(\\thetaZ_lm) = exp(-i [Y_l Y_m + X_l X_m] \\theta_lm)
    '''
    def __init__(self, Lx, Ly, n_lm_neighbors):
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
        self._refresh_uniraties_derivatives()
        return

    def _refresh_uniraties_derivatives(self):
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


    def _unpack_parameters(self, parameters):
        return parameters

    def _pack_parameters(self):
        return self.params

    def get_pairwise_distances(self):
        distances = np.zeros((self.Lx * self.Ly, self.Lx * self.Ly))

        for i in range(self.Lx * self.Ly):
            for j in range(self.Lx * self.Ly):
                xi, yi = i % self.Lx, i // self.Lx
                xj, yj = j % self.Lx, j // self.Lx

                '''
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
                '''

                distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)  # OBC
                distances[i, j] = distance

        assert np.allclose(distances, distances.T)

        return np.around(distances, decimals=4)

    def _get_matrices_locs(self):
        n_qubits = self.Lx * self.Ly
        matrices = []
        locs = []

        #for i in range(len(self.i_sites)):
        #    locs.append((self.i_sites[i],))
        #    matrices.append(sx)

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

        return matrices, locs

    def _get_derivative_idx(self, param_idx):
        return self.derivatives[param_idx]


    def _initialize_parameters(self):
        return np.random.uniform(size=len(self.locs))

    def set_parameters(self, parameters):
        self.params = parameters.copy()
        self._refresh_uniraties_derivatives()
        return

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