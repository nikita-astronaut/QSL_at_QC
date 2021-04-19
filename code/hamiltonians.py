import numpy as np
import scipy as sp
from scipy import sparse
import lattice_symmetries as ls
import utils

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


SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
P_ij = (SS + np.eye(4)) / 2.

SSun = -np.kron(sx, sx) + -np.kron(sy, sy) + np.kron(sz, sz)
P_ijun = (SSun + np.eye(4)) / 2.


class Hamiltonian(object):
    def __init__(self, basis, unitary, n_qubits, su2, symmetries, permutations, sectors, spin, **kwargs):
        self.n_qubits = n_qubits
        #self.basis = basis
        self.unitary = unitary
        self.symmetries = symmetries
        self.permutations = permutations
        self.sectors = sectors
        self.spin = spin
   
        ### obtaining ground state in the correct symmetry sector ###
        self.basis = ls.SpinBasis(ls.Group([ls.Symmetry(s, sector=sec) for s, sec in zip(self.symmetries, self.sectors)]), \
                                            number_spins=n_qubits, hamming_weight=n_qubits // 2 + self.spin if su2 else None)
        self.basis.build()
        

        self._matrix, self._terms, self.bonds = self._get_Hamiltonian_matrix(**kwargs)

        energy, ground_state = ls.diagonalize(self._matrix, k = 4, dtype=np.float64)
        print(energy)
        for idx, state in enumerate(ground_state.T):
            print('state', idx)
            for s in self.permutations:
                print(np.dot(state.conj(), state[s]))
        ### DEBUG
        
        print(energy - self.energy_renorm)
        spins = []
        all_bonds = []
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i != j:
                    all_bonds.append((i, j))
        total_spin = ls.Operator(self.basis, [ls.Interaction(SS, all_bonds)])        
        for s in ground_state.T:
            print(np.dot(s.conj(), total_spin(s)) + 3. * self.n_qubits)
            spins.append(np.dot(s.conj(), total_spin(s)) + 3. * self.n_qubits)
        
        ### END DEBUG
        for idx, s in enumerate(spins):
            if np.isclose(s / 4, self.spin * (self.spin + 1)):
                break
        print('idx = ', idx)

        ### rewrite ground state in terms of non-symmetrized basis ###
        gs_nonsymm = np.zeros(basis.number_states, dtype=np.complex128)
        gs_symm = ground_state[:, idx]  # FIXME
        for i in range(basis.number_states):
            nonsymm_state = basis.states[i]
            rep, character, norm = self.basis.state_info(nonsymm_state)
            if norm != 0.:
                gs_nonsymm[i] = gs_symm[self.basis.index(rep)] * norm * character
        self.ground_state = gs_nonsymm[np.newaxis, :]
        assert np.isclose(np.dot(gs_nonsymm.conj(), gs_nonsymm), 1.0)

        ### finally obtaining the GS in the nonsymmetric basis (provided from config) ###
        self.basis = basis
        self._matrix, self._terms, self.bonds = self._get_Hamiltonian_matrix(**kwargs)   

        assert np.isclose(np.dot(self._matrix(gs_nonsymm).conj(), gs_nonsymm), energy[idx])

        #self.ground_state = ground_state.T
        self.nterms = len(self._terms)
        print('ground state energy:', energy[0] - self.energy_renorm)
        print('system gap =', energy[1] - energy[0])
        self.gse = energy[0]
        #exit(-1)
        return

    def __call__(self, bra, n_term = None):
        if n_term is None:
            return self._matrix(bra)
        return self._terms[n_term][0](bra), self._terms[n_term][1]


    def _get_Hamiltonian_matrix(self, **kwargs):
        raise NotImplementedError()


class HeisenbergSquareNNBipartiteOBC(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = -1., j_zz = 1.):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems 
        assert Ly % 2 == 0

        operator = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
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
        return ls.Operator(self.basis, [ls.Interaction(operator, bonds)]), [ls.Operator(self.basis, [ls.Interaction(operator, [bond])]) for bond in bonds]


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
        return ls.Operator(self.basis, [ls.Interaction(operator, bonds)]), [ls.Operator(self.basis, [ls.Interaction(operator, [bond])]) for bond in bonds]

class HeisenbergSquare(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = +1., j_zz = 1., j2=0., BC='PBC'):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems
        assert Ly % 2 == 0

        operator = P_ij
        operator_j2 = P_ij
        operatorun = P_ijun
        operator_j2un = P_ijun

        n_sites = Lx * Ly

        bonds = []
        bonds_j2 = []
        bondsun = []
        bonds_j2un = []

        for site in range(n_sites):
            x, y = site % Lx, site // Lx

            site_up = ((x + 1) % Lx) + y * Lx
            site_right = x + ((y + 1) % Ly) * Lx

            if x + 1 < Lx or BC == 'PBC':
                if self.unitary[site, site_up] == +1:
                    bonds.append((site, site_up))
                else:
                    bondsun.append((site, site_up))
            if y + 1 < Ly or BC == 'PBC':
                if self.unitary[site, site_right] == +1:
                    bonds.append((site, site_right))
                else:
                    bondsun.append((site, site_right))


            site_up = ((x + 1) % Lx) + ((y + 1) % Ly) * Lx
            site_right = ((x + 1) % Lx) + ((y - 1) % Ly) * Lx
            if (x + 1 < Lx and y + 1 < Ly) or BC == 'PBC':
                if self.unitary[site, site_up] == +1:
                    bonds_j2.append((site, site_up))
                else:
                    bonds_j2un.append((site, site_up))
            if (x + 1 < Lx and y - 1 >= 0) or BC == 'PBC':
                if self.unitary[site, site_right] == +1:
                    bonds_j2.append((site, site_right))
                else:
                    bonds_j2un.append((site, site_right))

        self.energy_renorm = len(bonds) + len(bondsun) + len(bonds_j2) * j2 + len(bonds_j2un) * j2
        return ls.Operator(self.basis, ([ls.Interaction(operator * 2, bonds)] if len(bonds) > 0 else []) + \
                                       ([ls.Interaction(j2 * operator_j2 * 2, bonds_j2)] if len(bonds_j2) > 0 else []) + \
                                       ([ls.Interaction(operatorun * 2, bondsun)] if len(bondsun) > 0 else []) + \
                                       ([ls.Interaction(j2 * operator_j2un * 2, bonds_j2un)] if len(bonds_j2un) > 0 else [])), \
               ([[ls.Operator(self.basis, [ls.Interaction(operator, [bond])]), 2] for bond in bonds] if len(bonds) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operator_j2, [bond])]), j2 * 2.] for bond in bonds_j2] if len(bonds_j2) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operatorun, [bond])]), 2] for bond in bondsun] if len(bondsun) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operator_j2un, [bond])]), j2 * 2.] for bond in bonds_j2un] if len(bonds_j2un) > 0 else []), \
               bonds + bondsun + bonds_j2 + bonds_j2un


class HeisenbergHexagon(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = +1., j_zz = 1., j2=0., BC='PBC'):
        operator = P_ij
        operator_j2 = P_ij
        n_sites = 6

        bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        bonds_j2 = [(0, 2), (2, 4), (4, 0), (1, 3), (3, 5), (1, 5)]
        bondsun = []
        bonds_j2un = []

        self.energy_renorm = len(bonds) + len(bondsun) + len(bonds_j2) * j2 + len(bonds_j2un) * j2
        return ls.Operator(self.basis, ([ls.Interaction(operator * 2, bonds)] if len(bonds) > 0 else []) + \
                                       ([ls.Interaction(j2 * operator_j2 * 2, bonds_j2)] if len(bonds_j2) > 0 else []) + \
                                       ([ls.Interaction(operatorun * 2, bondsun)] if len(bondsun) > 0 else []) + \
                                       ([ls.Interaction(j2 * operator_j2un * 2, bonds_j2un)] if len(bonds_j2un) > 0 else [])), \
               ([[ls.Operator(self.basis, [ls.Interaction(operator, [bond])]), 2] for bond in bonds] if len(bonds) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operator_j2, [bond])]), j2 * 2.] for bond in bonds_j2] if len(bonds_j2) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operatorun, [bond])]), 2] for bond in bondsun] if len(bondsun) > 0 else []) + \
               ([[ls.Operator(self.basis, [ls.Interaction(operator_j2un, [bond])]), j2 * 2.] for bond in bonds_j2un] if len(bonds_j2un) > 0 else []), \
               bonds + bondsun + bonds_j2 + bonds_j2un

