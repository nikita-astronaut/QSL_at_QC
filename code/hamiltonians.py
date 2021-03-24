import numpy as np
import scipy as sp
from scipy import sparse
import lattice_symmetries as ls

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



class Hamiltonian(object):
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=n_qubits, hamming_weight=None)
        self.basis.build()

        self._matrix = self._get_Hamiltonian_matrix(**kwargs)

        energy, ground_state = ls.diagonalize(self._matrix, k = 2, dtype=np.float64)
        print('ground state energy:', energy[0])
        print('system gap =', energy[1] - energy[0])
        print('ground state vector:', ground_state[:, 0])
        return

    def __call__(self, bra):
        return self._matrix(bra)

    def _get_Hamiltonian_matrix(self, **kwargs):
        raise NotImplementedError()


class HeisenbergSquareNNBipartiteOBC(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = -1., j_zz = 1.):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems 
        assert Ly % 2 == 0

        operator = j_pm * (np.kron(sx, sx) + np.kron(sy, sy)) + j_zz * np.kron(sz, sz)
        n_sites = Lx * Ly

        bonds = []
        for site in range(n_sites):
            x, y = site % Lx, site // Lx

            site_up = ((x + 1) % Lx) + y * Lx
            site_right = x + ((y + 1) % Ly) * Lx

            if x + 1 < Lx:
                bonds.append((site, site_up))
            if y + 1 < Ly:
                bonds.append((site, site_right))
        print('bonds = ', bonds)
        return ls.Operator(self.basis, [ls.Interaction(operator, bonds)])


class HeisenbergSquareNNBipartitePBC(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = +1., j_zz = 1.):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems 
        assert Ly % 2 == 0

        operator = j_pm * (np.kron(sx, sx) + np.kron(sy, sy)) + j_zz * np.kron(sz, sz)
        n_sites = Lx * Ly

        bonds = []
        for site in range(n_sites):
            x, y = site % Lx, site // Lx

            site_up = ((x + 1) % Lx) + y * Lx
            site_right = x + ((y + 1) % Ly) * Lx
            bonds.append((site, site_up))
            bonds.append((site, site_right))
        print('bonds = ', bonds)
        return ls.Operator(self.basis, [ls.Interaction(operator, bonds)])

class HeisenbergSquareNNNBipartitePBC(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = +1., j_zz = 1., j2=0.):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems
        assert Ly % 2 == 0

        operator = j_pm * (np.kron(sx, sx) + np.kron(sy, sy)) + j_zz * np.kron(sz, sz)
        operator_j2 = j2 * (j_pm * (np.kron(sx, sx) + np.kron(sy, sy)) + j_zz * np.kron(sz, sz))
        n_sites = Lx * Ly

        bonds = []
        bonds_j2 = []
        for site in range(n_sites):
            x, y = site % Lx, site // Lx

            site_up = ((x + 1) % Lx) + y * Lx
            site_right = x + ((y + 1) % Ly) * Lx
            bonds.append((site, site_up))
            bonds.append((site, site_right))

            site_up = ((x + 1) % Lx) + ((y + 1) % Ly) * Lx
            site_right = ((x + 1) % Lx) + ((y - 1) % Ly) * Lx
            bonds_j2.append((site, site_up))
            bonds_j2.append((site, site_right))

        return ls.Operator(self.basis, [ls.Interaction(operator, bonds), ls.Interaction(operator_j2, bonds_j2)])

