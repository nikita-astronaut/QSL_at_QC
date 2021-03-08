import numpy as np
import qutip as qtp
import scipy as sp


sz = np.array([[1, 0], \
               [0, -1]])

sx = np.array(
               [[0, 1], \
               [1, 0]]
              )

sy = np.array([[0, 1.0j], [-1.0j, 0]])

s0 = np.eye(2)
sx_sparse = sp.sparse.csr_matrix(sx)
sy_sparse = sp.sparse.csr_matrix(sy)
sz_sparse = sp.sparse.csr_matrix(sz)
s0_sparse = sp.sparse.csr_matrix(s0)



def get_SS_gate(i, j, n_qubits, pauli_sparse, phase):
    SS = sp.sparse.eye(1, dtype=np.complex128)

    for k in range(n_qubits):
        SS = scipy.sparse.kron(SS, pauli_sparse if k in [i, j] else s0_sparse)

    return sp.sparse.linalg.expm(SS * phase)

def get_S_gate(i, n_qubits, pauli_sparse, phase):
    S = sp.sparse.eye(1, dtype=np.complex128)

    for k in range(n_qubits):
        S = scipy.sparse.kron(S, pauli_sparse if k == i else s0_sparse)

    return sp.sparse.linalg.expm(S * phase)


class Circuit(object):
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        self._unitary = self._get_unitary_matrix(**kwargs)

    def __call__(self):
        state_alldown = np.zeros(2 ** n_qubits, dtype=np.complex128)
        state_alldown[0] = 1.

        return self._unitary.dot(state_alldown)

    def _define_parameter_site_bonds(self, **kwargs):
        raise NotImplementedError()

    def _get_unitary_matrix(self, **kwargs):
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

        self.pairwise_distances = self.get_pairwise_distances()
        self.theta_i_sites, self.theta_XY_lm_bonds, self.theta_Z_lm_bonds = self._define_parameter_site_bonds()

        self.theta_i, self.theta_XY_lm, self.theta_Z_lm = self._initialize_parameters()
        return super().__init__(Lx * Ly, self.theta_i, self.theta_XY_lm, self.theta_Z_lm)

    def _define_parameter_site_bonds(self):
        theta_i_sites = np.arange(self.Lx * self.Ly)
        theta_XY_lm_bonds = []
        theta_Z_lm_bonds = []

        unique_distances = np.sort(np.unique(self.pairwise_distances.flatten()))


        for l in range(self.Lx * self.Ly):
            for m in range(l + 1, self.Lx * self.Ly):
                dist = self.pairwise_distances[l, m]

                separation = np.where(unique_distances == dist)[0][0]

                assert separation > 0

                if separation >= self.n_lm_neighbors:
                    continue

                theta_XY_lm_bonds.append((l, m))
                theta_Z_lm_bonds.append((l, m))

        return theta_i_sites, theta_XY_lm_bonds, theta_Z_lm_bonds


    def _unpack_parameters(self, parameters):
        return parameters[:len(self.theta_i_sites)], \
               parameters[len(self.theta_i_sites):-len(self.theta_Z_lm)], \
               parameters[-len(self.theta_Z_lm):]

    def _pack_parameters(self):
        return np.concatenate([self.theta_i, self.theta_XY_lm, self.theta_Z_lm], axis = 0)

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
                distances[i, j] = distance

        assert np.allclose(distances, distances.T)

        return np.around(distances, decimals=4)


    def _get_unitary_matrix(self):
        n_qubits = self.Lx * self.Ly
        unitary = sp.sparse.eye(2 ** n_qubits, dtype=np.complex128)
        for ti, i in zip(self.theta_i, self.theta_i_sites):
            unitary = unitary.dot(get_S_gate(i, n_qubits, sz_sparse, ti))

        for t_perp_lm, t_Z_lm, bond in zip(self.theta_perp_lm, self.theta_Z_lm, self.theta_Z_lm_bonds):
            unitary = unitary.dot(get_SS_gate(i, j, n_qubits, sx_sparse, t_perp_lm))
            unitary = unitary.dot(get_SS_gate(i, j, n_qubits, sy_sparse, t_perp_lm))
            unitary = unitary.dot(get_SS_gate(i, j, n_qubits, sz_sparse, t_Z_lm))
        return unitary


    def _initialize_parameters(self):
        theta_i = []
        theta_XY_lm = np.zeros(len(self.theta_XY_lm_bonds), dtype=np.float64)
        theta_Z_lm = np.zeros(len(self.theta_Z_lm_bonds), dtype=np.float64)

        for x in range(self.Lx):
            for y in range(self.Ly):
                theta_i.append(np.pi * (-1) ** (x + y))
        return np.array(theta_i), theta_XY_lm, theta_Z_lm

    def set_parameters(self, parameters):
        self.theta_i, self.theta_XY_lm, self.theta_Z_lm = self._unpack_parameters(parameters)
        self._unitary = self._get_unitary_matrix()

        return
