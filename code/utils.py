import os
import sys
import numpy as np
from time import time
import lattice_symmetries as ls
import qiskit

def index_to_spin(index, number_spins = 16):
    return (((index.reshape(-1, 1).astype(np.int64) & (1 << np.arange(number_spins).astype(np.int64)))) > 0)

def spin_to_index(spin, number_spins = 16):
    a = 2 ** np.arange(number_spins)
    return spin.dot(a)

def last_to_ij(i, j, number_spins = 16):
    idxs = np.arange(2 ** number_spins)
    spin = index_to_spin(idxs, number_spins)
    spin[:, i], spin[:, N - 2] = spin[:, N - 2], spin[:, i]
    spin[:, j], spin[:, N - 1] = spin[:, N - 1], spin[:, j]
    return spin_to_index(spin, number_spins)

def import_config(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the module from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the module from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module


def get_x_symmetry_map(Lx, Ly, basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = (x + 1) % Lx + y * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = Lx * Ly))

def get_y_symmetry_map(Lx, Ly, basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = x + ((y + 1) % Ly) * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = Lx * Ly))



def get_Cx_symmetry_map(Lx, Ly, basis, su2=False):
    n_qubits = Lx * Ly
    spins = index_to_spin(basis.states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = (Lx - 1 - x) % Lx + y * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = Lx * Ly))


def r_to_index(x, y, Lx, Ly):
    return y * Lx + x

def index_to_r(idx, Lx, Ly):
    return idx % Lx, idx // Lx


def get_rot_symmetry_map(Lx, Ly, basis, su2=False):
    assert Lx == Ly
    assert Lx == 4
    '''
    xmap = []
    L = Lx
    for idx in range(L ** 2):
        x, y = index_to_r(idx, L, L)
        xpr = (-y) % L
        ypr = x

        xmap.append(r_to_index(xpr, ypr, L, L))

    
    xmap = np.array(xmap)
    '''

    xmap = np.array([12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3])
    n_qubits = Lx * Ly
    spins = index_to_spin(basis.states, number_spins = Lx * Ly)

    spins = spins[:, xmap]

    return xmap, np.argsort(spin_to_index(spins, number_spins = Lx * Ly))


def get_Cy_symmetry_map(Lx, Ly, basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = x % Lx + ((Ly - 1 - y) % Ly) * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = Lx * Ly))

def get_hexagon_rot_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 6)

    map_site = np.array([1, 2, 3, 4, 5, 0])
    assert np.allclose(map_site[map_site][map_site][map_site][map_site][map_site], np.arange(6))
  

    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 6))

def get_hexagon_mir_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 6)

    map_site = np.array([0, 5, 4, 3, 2, 1])
    assert np.allclose(map_site[map_site], np.arange(6))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 6))


def get_honeycomb_2x2_trx_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 8)

    map_site = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    assert np.allclose(map_site[map_site], np.arange(8))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 8))


def get_honeycomb_3x3_trx_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 18)

    map_site = np.array([2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 6, 7, 14, 15, 16, 17, 12, 13])
    assert np.allclose(map_site[map_site][map_site], np.arange(18))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 18))


def get_honeycomb_2x2_try_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 8)

    map_site = np.array([2, 3, 0, 1, 6, 7, 4, 5])
    assert np.allclose(map_site[map_site], np.arange(8))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 8))


def get_honeycomb_3x3_try_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 18)

    map_site = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5])
    assert np.allclose(map_site[map_site][map_site], np.arange(18))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 18))



def get_honeycomb_2x2_rot_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 8)

    map_site = np.array([1, 4, 7, 2, 3, 6, 5, 0])
    assert np.allclose(map_site[map_site][map_site][map_site][map_site][map_site], np.arange(8))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 8))


def get_honeycomb_3x3_rot_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 18)

    map_site = np.array([1, 6, 15, 2, 11, 16, 3, 8, 17, 4, 7, 12, 5, 10, 13, 0, 9, 14])
    assert np.allclose(map_site[map_site][map_site][map_site][map_site][map_site], np.arange(18))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 18))


def get_honeycomb_2x2_mir1_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 8)

    map_site = np.array([0, 5, 6, 3, 4, 1, 2, 7])
    assert np.allclose(map_site[map_site], np.arange(8))


    spins = spins[:, map_site] 
    return map_site, np.argsort(spin_to_index(spins, number_spins = 8))


def get_honeycomb_3x3_mir2_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 18)

    map_site = np.array([0, 13, 14, 9, 10, 5, 12, 7, 8, 3, 4, 17, 6, 1, 2, 15, 16, 11])
    assert np.allclose(map_site[map_site], np.arange(18))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 18))

def get_honeycomb_3x3_mir1_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 18)

    map_site = np.array([17, 16, 13, 12, 15, 14, 7, 6, 9, 8, 11, 10, 3, 2, 5, 4, 1, 0])
    assert np.allclose(map_site[map_site], np.arange(18))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 18))


def get_honeycomb_2x2_mir2_symmetry_map(basis, su2=False):
    spins = index_to_spin(basis.states, number_spins = 8)

    map_site = np.array([7, 2, 1, 4, 3, 6, 5, 0])
    assert np.allclose(map_site[map_site], np.arange(8))


    spins = spins[:, map_site]
    return map_site, np.argsort(spin_to_index(spins, number_spins = 8))


def compute_norm_sample(state, projector, N_samples):
    t = time()
    norms = []
    #state_ancilla = np.zeros(2 * len(state), dtype=np.complex128)
    #indexes = np.arange(len(state_ancilla))


    ps = [(0.5 - np.vdot(projector(state, proj_idx), state) / 2.).real for proj_idx in range(projector.nterms)]
    ps = np.clip(ps, a_min=0., a_max=1.)
    N_ups = np.random.binomial(N_samples, ps)
    print('sample estimation of N(theta) = ', time() - t)
    return np.sum(1 - 2 * N_ups / N_samples) / projector.nterms

    for proj_idx in range(projector.nterms):
        state_proj = projector(state, proj_idx)
        state_ancilla[:len(state)] = (state + state_proj) / 2.
        state_ancilla[len(state):] = (state - state_proj) / 2.
        

        idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)

        N_up = np.sum(idxs > len(state))

        norms.append(1 - 2 * N_up / N_samples)

    print('sample estimation of N(theta) = ', time() - t)
    return np.sum(norms) / projector.nterms


from copy import deepcopy
def get_symmetry_unique_bonds(bonds, permutations):
    removed_bonds = []
    groups = []
    idxs_groups = []
    tot_size = 0

    for bond in bonds:
        group = []
        idxs_group = []
        bonds_symm = [(perm[bond[0]], perm[bond[1]]) for perm in permutations]
        
        for bond_symm in bonds_symm:
            if bond_symm in removed_bonds or bond_symm[::-1] in removed_bonds:
                continue

            if bond_symm in bonds:
                removed_bonds.append(bond_symm)
                group.append(bond_symm)
                idxs_group.append(bonds.index(bond_symm))
            else:
                removed_bonds.append(bond_symm[::-1])
                group.append(bond_symm[::-1])
                idxs_group.append(bonds.index(bond_symm[::-1]))
        if len(group) == 0:
            continue
        groups.append(deepcopy(group))
        idxs_groups.append(deepcopy(idxs_group))
        tot_size += len(group)
    assert tot_size == len(bonds)
    return groups, idxs_groups




def compute_energy_sample(state, hamiltonian, projector, N_samples):
    t = time()

    state0_proj_inv = np.array([projector(state, proj_idx, inv=True) for proj_idx in range(projector.nterms)]).conj()

    js = np.array([hamiltonian(None, ham_idx, vector=False) for ham_idx in range(hamiltonian.nterms)])
    states_ham = np.array([hamiltonian(state, ham_idx) for ham_idx in range(hamiltonian.nterms)])

    ps = (0.5 - np.dot(state0_proj_inv, states_ham.T) / 2.).real
    N_ups = np.random.binomial(N_samples, ps)
    energy = np.sum((1 - 2 * N_ups / N_samples).dot(js))
    print('sample estimation of E(theta) = ', time() - t)
    return energy / projector.nterms


def compute_energy_sample_symmetrized(state, hamiltonian, projector, N_samples):
    bond_groups, idxs_groups = get_symmetry_unique_bonds(hamiltonian.bonds, projector.maps)

    t = time()
    energies = []
    state_ancilla = np.zeros(2 * len(state), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))
    
    for idxs_group in idxs_groups:
        ham_idx = idxs_group[0]
        state_ham, j = hamiltonian(state, ham_idx)
        for proj_idx in range(projector.nterms):
            state_proj = projector(state_ham, proj_idx)
            state_ancilla[:len(state)] = (state + state_proj) / 2.
            state_ancilla[len(state):] = (state - state_proj) / 2.

            idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples * 5)
  
            N_up = np.sum(idxs >= len(state))

            energies.append((1 - 2 * N_up / N_samples / 5.) * j * len(idxs_group))

    print('sample estimation of E(theta) = ', time() - t)
    return np.sum(energies) / projector.nterms


def compute_metric_tensor_sample(states, projector, N_samples):
    t = time()
    t_proj = 0.
    t_norm = 0.
    t_samp = 0.

    ctr = 0
    ps_real = np.empty((len(states), len(states), projector.nterms), dtype=np.float64)
    ps_imag = np.empty((len(states), len(states), projector.nterms), dtype=np.float64)
    
    for proj_idx in range(projector.nterms):
        z = time()
        states_j_proj = projector(states, proj_idx).conj() # np.array([projector(states[j], proj_idx) for j in range(len(states))]).conj(
        t_proj += time() - z

        z = time()
        ps = 0.5 - np.dot(states, states_j_proj.T).conj() / 2.
        ps_real[..., proj_idx] = ps.real
        ps_imag[..., proj_idx] = ps.imag + 0.5

        t_norm += time() - z

    z = time()
    N_ups_reals = np.random.binomial(N_samples, np.clip(ps_real, a_min=0., a_max=1.))
    N_ups_imags = np.random.binomial(N_samples, np.clip(ps_imag, a_min=0., a_max=1.))
    t_samp += time() - z


    MT = (1 - 2 * N_ups_reals / N_samples).mean(axis=-1) + 1.0j * (1 - 2 * N_ups_imags / N_samples).mean(axis=-1)
    print('sample estimation of MT(theta) = ', time() - t)
    print(t_proj, t_norm, t_samp)
    return MT


def compute_connectivity_sample(state0, states, projector, N_samples, theta=0.):
    t = time()
    thetafull = np.exp(1.0j * np.pi / 2. * theta)

    state0_proj_inv = np.array([projector(state0, proj_idx, inv=True) / thetafull for proj_idx in range(projector.nterms)]).conj()


    ps = (0.5 - np.dot(state0_proj_inv, states.T) / 2.).real
    N_ups = np.random.binomial(N_samples, ps)
    connectivity = np.sum((1 - 2 * N_ups / N_samples), axis = 0)

    print('sample estimation of connectivity(theta) = ', time() - t)
    return connectivity / projector.nterms

def compute_energy_der_sample(state0, states, hamiltonian, projector, N_samples, theta=0.):
    t = time()
    time_sampling = 0.
    t_proj = 0.
    t_ham = 0.
    t_samp = 0.
    t_norm = 0.

    z = time()
    state0_proj_inv = np.array([projector(state0, proj_idx, inv=True) for proj_idx in range(projector.nterms)]).conj()
    t_proj += time() - z


    ps = np.empty((projector.nterms, states.shape[1], hamiltonian.nterms), dtype=np.float64)
    js = []
    for ham_idx in range(hamiltonian.nterms):
        z = time()
        states_ham = hamiltonian(states, ham_idx) # np.empty((hamiltonian.nterms, len(state0)), dtype=np.complex128)
        j = hamiltonian(None, ham_idx, vector=False)
        js.append(j)
        t_ham += time() - z

        z = time()
        ps[..., ham_idx] = 0.5 - np.dot(state0_proj_inv, states_ham).real / 2.
        t_norm += time() - z

    z = time()
    N_ups = np.random.binomial(N_samples, ps)
    t_samp += time() - z

    der = np.sum((1 - 2 * N_ups / N_samples).dot(js), axis=0)

    print(t_proj, t_ham, t_samp, t_norm)
    print('sample estimation of energy der numerator(theta) = ', time() - t, 'sampling:', time_sampling)
    return der / projector.nterms


def compute_norm_qiskit(circuit, projector, N_samples, noise_model):
    basis_gates = noise_model.basis_gates

    circ = circuit.init_circuit_qiskit()
    result = qiskit.execute(circ, qiskit.Aer.get_backend('statevector_simulator'), shots=2 ** 14).result()
    print(repr(result.get_statevector(circ)))

    circ = circuit.act_dimerization_qiskit(circ)
    #result = qiskit.execute(circ, qiskit.Aer.get_backend('statevector_simulator'), shots=2 ** 14).result()
    #return result.get_statevector(circ)
    circ = circuit.act_psi_qiskit(circ, circuit.params)

    projector_circuits = []
    for idx, cycl in enumerate(projector.cycl):
        current_circ = circ.copy('circuit_projector_{:d}'.format(idx))

        pair_permutations = []
        for loop in cycl:
            if len(loop) == 1:
                continue

            for i in range(1, len(loop)):
                pair_permutations.append((loop[0], loop[i]))


        current_circ = circuit.act_permutation_qiskit(current_circ, pair_permutations)

        #result = qiskit.execute(current_circ, qiskit.Aer.get_backend('statevector_simulator'), shots=2 ** 14).result()
        #return result.get_statevector(current_circ)

        current_circ = circuit.act_psi_qiskit(current_circ, circuit.params, inverse=True)
        current_circ = circuit.act_dimerization_qiskit(current_circ, inverse=True)
        current_circ.measure(np.arange(circuit.n_qubits), np.arange(circuit.n_qubits))

        projector_circuits.append(current_circ.copy())

    t = time()
    result = qiskit.execute(projector_circuits, qiskit.Aer.get_backend('qasm_simulator'), shots=2 ** 4,
                            basis_gates=None,
                            noise_model=noise_model).result()
    print('noisy simulation with quasm took', time() - t)

    #print(repr(result.get_statevector(projector_circuits[0])))
    survivals = []
    for i in range(len(projector.cycl)):
        res = result.get_counts(i).get('0' * circuit.n_qubits)
        survivals.append(res if res is not None else 0)
    print(survivals)
    return np.mean(np.sqrt(np.array(survivals) / 2 ** 4))


