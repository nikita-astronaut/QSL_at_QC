import numpy as np 
import lattice_symmetries as ls
import utils
from copy import deepcopy

class Projector(object):
    def __call__(self, state, n_term = None, inv=False):
        if n_term is not None:
            if inv:
                return state[..., self.permutations_inv[n_term]] * 1. / self.characters[n_term] if self.characters[n_term] != 1.0 else state[..., self.permutations_inv[n_term]]
            return state.T[self.permutations[n_term], ...].T * self.characters[n_term] if self.characters[n_term] != 1.0 else np.ascontiguousarray(state.T)[self.permutations[n_term], ...].T

        state_projected = state * 0.0
        for permutation, character in zip(self.permutations, self.characters):
            state_projected += state[..., permutation] * character
        return state_projected / len(self.permutations)
        

class ProjectorFull(Projector):
    def __init__(self, n_qubits, su2, basis, generators, eigenvalues, degrees):
        self.basis = basis
        self.n_qubits = n_qubits
        self.basis_size = basis.number_states

        self.maps, self.permutations, self.characters, self.cycl = self._init_projector(generators, eigenvalues, degrees)
        self.permutations_inv = [np.argsort(perm) for perm in self.permutations]
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

        for idx, p in enumerate(permutations):
            for k in range(idx + 1, len(permutations)):
                if np.allclose(p, permutations[k]):
                    print('coincide', idx, k, p)
                    total -= 1
        print('unique', total)


        def cycles(perm):
            remain = set(perm)
            result = []
            while len(remain) > 0:
                n = remain.pop()
                cycle = [n]
                while True:
                    n = perm[n]
                    if n not in remain:
                        break
                    remain.remove(n)
                    cycle.append(n)
                result.append(cycle)
            return result


        #for idx, p in enumerate(maps):
        #    print(p,  sum([len(l) - 1 for l in cycles(p)]))

        cycl = []
        for idx, p in enumerate(maps):
            map_swaps = []
            cycl.append(cycles(p))

        return maps, permutations, characters, cycl



