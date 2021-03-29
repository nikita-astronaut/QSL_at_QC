import numpy as np 
import lattice_symmetries as ls
import utils
from copy import deepcopy

class Projector(object):
    def __call__(self, state, n_term = None):
        if n_term is not None:
            return state[self.permutations[n_term]]

        state_projected = state * 0.0
        for permutation, character in zip(self.permutations, self.characters):
            state_projected += state[permutation] * character
        return state_projected / len(self.permutations)
        

class ProjectorFull(Projector):
    def __init__(self, n_qubits, su2, generators, eigenvalues, degrees):
        self.n_qubits = n_qubits
        basis = ls.SpinBasis(ls.Group([]), number_spins=n_qubits, hamming_weight=n_qubits // 2 if su2 else None)#, spin_inversion=-1 if su2 else 0)
        basis.build() 
        self.basis_size = basis.number_states

        self.maps, self.permutations, self.characters = self._init_projector(generators, eigenvalues, degrees)
        self.nterms = len(self.permutations)


        return

    def _init_projector(self, generators, eigenvalues, degrees):
        permutations = [np.arange(self.basis_size)]
        maps = [np.arange(self.n_qubits)]
        characters = [1. + 0.0j]

        if len(generators) == 0:
            return maps, permutations, characters

        for generator, eigenvalue, degree in zip(generators, eigenvalues, degrees):
            m, g = generator
            m = m.copy()
            g = g.copy()

            lamb = eigenvalue

            new_permutations = []
            new_characters = []
            new_maps = []

            for d in range(degree - 1):
                permutations_d = []
                characters_d = []
                maps_d = []
                for mm, symm, ch in zip(maps, permutations, characters): 
                    permutations_d.append(symm[g])
                    characters_d.append(lamb * ch)
                    maps_d.append(mm[m])

                new_permutations.append(deepcopy(permutations_d))
                new_characters.append(deepcopy(characters_d))
                new_maps.append(deepcopy(maps_d))

                lamb *= eigenvalue
                g = g[generator[1]]
                m = m[generator[0]]

            for perm in new_permutations:
                permutations += perm
            for ch in new_characters:
                characters += ch
            for x in new_maps:
                maps += x


        total = 1
        for d in degrees:
            total *= d


        assert len(permutations) == total

        print('terms in the projector:', total)
        return maps, permutations, characters



