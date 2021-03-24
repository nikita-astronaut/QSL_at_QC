import os
import sys
import numpy as np

def index_to_spin(index, number_spins = 16):
    return (((index.reshape(-1, 1) & (1 << np.arange(number_spins)))) > 0)

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


def get_x_symmetry_map(Lx, Ly):
    states = np.arange(2 ** (Lx * Ly), dtype=np.int64)
    spins = index_to_spin(states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = (x + 1) % Lx + y * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return spin_to_index(spins, number_spins = Lx * Ly)

def get_y_symmetry_map(Lx, Ly):
    states = np.arange(2 ** (Lx * Ly), dtype=np.int64)
    spins = index_to_spin(states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = x + ((y + 1) % Ly) * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return spin_to_index(spins, number_spins = Lx * Ly)


def get_Cx_symmetry_map(Lx, Ly):
    states = np.arange(2 ** (Lx * Ly), dtype=np.int64)
    spins = index_to_spin(states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = (Lx - 1 - x) % Lx + y * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return spin_to_index(spins, number_spins = Lx * Ly)

def get_Cy_symmetry_map(Lx, Ly):
    states = np.arange(2 ** (Lx * Ly), dtype=np.int64)
    spins = index_to_spin(states, number_spins = Lx * Ly)

    map_site = []
    for i in range(Lx * Ly):
        x, y = i % Lx, i // Lx
        j = x % Lx + ((Ly - 1 - y) % Ly) * Lx
        map_site.append(j)
    map_site = np.array(map_site)

    spins = spins[:, map_site]
    return spin_to_index(spins, number_spins = Lx * Ly)
