import numpy as np 
import lattice_symmetries as ls
import utils
from copy import deepcopy

class Projector(object):
    def __call__(self, state):
        state_projected = state * 0.0
        for permutation, character in zip(self.permutations, self.characters):
            state_projected += state[permutation] * character
        return state_projected / len(self.permutations)


class ProjectorFull(Projector):
    def __init__(self, n_qubits, generators, eigenvalues, degrees):
        self.n_qubits = n_qubits
        self.permutations, self.characters = self._init_projector(generators, eigenvalues, degrees)
        return

    def _init_projector(self, generators, eigenvalues, degrees):
        permutations = [np.arange(2 ** self.n_qubits)]
        characters = [1. + 0.0j]

        if len(generators) == 0:
            return permutations, characters

        for generator, eigenvalue, degree in zip(generators, eigenvalues, degrees):
            g = generator.copy()
            lamb = eigenvalue
            new_permutations = []
            new_characters = []

            for d in range(degree - 1):
                permutations_d = []
                characters_d = []
                for symm, ch in zip(permutations, characters): 
                    permutations_d.append(symm[g])
                    characters_d.append(lamb * ch)

                new_permutations.append(deepcopy(permutations_d))
                new_characters.append(deepcopy(characters_d))

                lamb *= eigenvalue
                g = g[generator]

            for perm in new_permutations:
                permutations += perm
            for ch in new_characters:
                characters += ch


        total = 1
        for d in degrees:
            total *= d


        assert len(permutations) == total

        return permutations, characters


