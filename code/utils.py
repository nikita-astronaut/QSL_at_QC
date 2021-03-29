import os
import sys
import numpy as np
from time import time
import lattice_symmetries as ls

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


def compute_norm_sample(state, projector, N_samples):
    t = time()
    norms = []
    state_ancilla = np.zeros(2 * len(state), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))

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
    energies = []
    state_ancilla = np.zeros(2 * len(state), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))

    for ham_idx in range(hamiltonian.nterms):
        state_ham, j = hamiltonian(state, ham_idx)
        for proj_idx in range(projector.nterms):
            state_proj = projector(state_ham, proj_idx)
            state_ancilla[:len(state)] = (state + state_proj) / 2.
            state_ancilla[len(state):] = (state - state_proj) / 2.

            idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)
            #np.random.seed(0)
            #idxs = np.random.randint(0, len(state_ancilla), size = N_samples)
 
            N_up = np.sum(idxs >= len(state))

            energies.append((1 - 2 * N_up / N_samples) * j)
            #print(energies[-1])
        #print('energy_projected: ', np.sum(energies[-projector.nterms:]), 'idx', ham_idx, j)

    print('sample estimation of E(theta) = ', time() - t)
    return np.sum(energies) / projector.nterms


def compute_energy_sample_symmetrized(state, hamiltonian, projector, N_samples):
    bond_groups, idxs_groups = get_symmetry_unique_bonds(hamiltonian.bonds, projector.maps)
    #print(bond_groups, idxs_groups)

    t = time()
    energies = []
    state_ancilla = np.zeros(2 * len(state), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))
    
    #np.random.seed(0)
    #idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)
    for idxs_group in idxs_groups:
        ham_idx = idxs_group[0]
        #print(idxs_group)
        state_ham, j = hamiltonian(state, ham_idx)
        for proj_idx in range(projector.nterms):
            state_proj = projector(state_ham, proj_idx)
            state_ancilla[:len(state)] = (state + state_proj) / 2.
            state_ancilla[len(state):] = (state - state_proj) / 2.

            idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples * 5)
            #np.random.seed(0)
            #idxs = np.random.randint(0, len(state_ancilla), size = N_samples)
  
            N_up = np.sum(idxs >= len(state))

            energies.append((1 - 2 * N_up / N_samples / 5.) * j * len(idxs_group))
            #print(energies[-1] / len(idxs_group))
        #print('energy_projected: ', np.sum(energies[-projector.nterms:]), 'idx', ham_idx, j)

    print('sample estimation of E(theta) = ', time() - t)
    return np.sum(energies) / projector.nterms


def compute_metric_tensor_sample(states, projector, N_samples, theta=0.):
    t = time()
    MT = np.zeros((len(states), len(states)), dtype=np.complex128)
    state_ancilla = np.zeros(2 * len(states[0]), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))

    for proj_idx in range(projector.nterms):
        for i in range(len(states)):
            for j in range(i, len(states)):
                state_j_proj = projector(states[j], proj_idx)
                state_ancilla[:len(state_j_proj)] = (states[i] + np.exp(1.0j * theta) * state_j_proj) / 2.
                state_ancilla[len(state_j_proj):] = (states[i] - np.exp(1.0j * theta) * state_j_proj) / 2.

                idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)
                N_up = np.sum(idxs >= len(states[0]))

                MT[i, j] += (1 - 2 * N_up / N_samples)
                if i != j and np.isclose(theta, 0.):
                    MT[j, i] += (1 - 2 * N_up / N_samples)
                if i != j and np.isclose(theta, -np.pi / 2):
                    MT[j, i] -= (1 - 2 * N_up / N_samples)

                if i == j and np.isclose(theta, -np.pi / 2.):
                    MT[i, i] = 0.

    print('sample estimation of MT(theta) = ', time() - t)
    return MT / projector.nterms


def compute_connectivity_sample(state0, states, projector, N_samples, theta=0.):
    t = time()
    connectivity = np.zeros(len(states), dtype=np.complex128)
    state_ancilla = np.zeros(2 * len(states[0]), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))

    for proj_idx in range(projector.nterms):
        for i in range(len(states)):
            state_i_proj = projector(states[i], proj_idx)
            state_ancilla[:len(state_i_proj)] = (state0 + np.exp(1.0j * theta) * state_i_proj) / 2.
            state_ancilla[len(state_i_proj):] = (state0 - np.exp(1.0j * theta) * state_i_proj) / 2.

            idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)
            N_up = np.sum(idxs >= len(state0))

            connectivity[i] += (1 - 2 * N_up / N_samples)

    print('sample estimation of connectivity(theta) = ', time() - t)
    return connectivity / projector.nterms

def compute_energy_der_sample(state0, states, hamiltonian, projector, N_samples, theta=0.):
    t = time()
    der = np.zeros(len(states), dtype=np.complex128)
    state_ancilla = np.zeros(2 * len(states[0]), dtype=np.complex128)
    indexes = np.arange(len(state_ancilla))
    time_sampling = 0.

    for i in range(len(states)):
        for ham_idx in range(hamiltonian.nterms):
            state_ham, j = hamiltonian(states[i], ham_idx)
            for proj_idx in range(projector.nterms):
                state_i_proj = projector(state_ham, proj_idx)
                state_ancilla[:len(state_i_proj)] = (state0 + np.exp(1.0j * theta) * state_i_proj) / 2.
                state_ancilla[len(state_i_proj):] = (state0 - np.exp(1.0j * theta) * state_i_proj) / 2.

                ts = time()
                idxs = np.random.choice(indexes, p=np.abs(state_ancilla) ** 2, replace=True, size=N_samples)
                time_sampling += time() - ts
                N_up = np.sum(idxs >= len(state0))

                der[i] += (1 - 2 * N_up / N_samples) * j

    print('sample estimation of energy der numerator(theta) = ', time() - t, 'sampling:', time_sampling)
    return der / projector.nterms
