import numpy at np
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



class Hamiltonian(object):
    def __init__(self, **kwargs):
        self._matrix = self._get_Hamiltonian_matrix(**kwargs)

    def __call__(self, bra):
        return self._matrix.dot(bra)

    def _get_Hamiltonian_matrix(self, **kwargs):
        raise NotImplementedError()


class HeisenbergSquareNNBipartiteSparse(Hamiltonian):
    def _get_Hamiltonian_matrix(self, Lx, Ly, j_pm = -1., j_zz = 1.):
        assert Lx % 2 == 0  # here we only ocnsider bipartite systems 
        assert Ly % 2 == 0

        n_sites = Lx * Ly

        bonds = []
        H = sp.sparse.csr_matrix(shape=(2 ** n_sites, 2 ** n_sites), dtype=np.complex128)
        for site in range(n_sites):
            x, y = site % Lx, site // Lx

            site_up = ((x + 1) % Lx) + y * Lx
            site_right = x + ((y + 1) % Ly) * Lx

            for bond in [(site, site_up), (site, site_right)]:
                h_xx = sp.sparse.eye(1, dtype=np.complex128)
                h_xx = sp.sparse.eye(1, dtype=np.complex128)
                h_xx = sp.sparse.eye(1, dtype=np.complex128)

                for i in range(n_sites):
                    h_xx = scipy.sparse.kron(h_xx, sx_sparse if i in bond else s0_sparse)
                    h_yy = scipy.sparse.kron(h_yy, sy_sparse if i in bond else s0_sparse)
                    h_zz = scipy.sparse.kron(h_zz, sz_sparse if i in bond else s0_sparse)

                H += j_pm * (h_xx + h_yy) + j_zz * h_zz

        assert np.isclose(sp.sparse.csr_matrix.sum(H - sp.sparse.csr_matrix.transpose(sp.sparse.csr_matrix.conjugate(H))), 0.0)

        return H.real


