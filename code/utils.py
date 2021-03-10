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


