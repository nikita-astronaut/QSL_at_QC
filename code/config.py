import numpy as np
import circuits
import hamiltonians
import optimizers
from scipy.optimize import differential_evolution

class opt_parameters:
    def __init__(self):
        self.Lx, self.Ly = 4, 4
        
        self.hamiltonian = hamiltonians.HeisenbergSquareNNBipartiteSparse;
        self.ham_params_dict = {'Lx' : self.Lx, 'Ly': self.Ly, 'j_pm' : -1., 'j_zz' : 1.}

        self.circuit = circuits.TrotterizedMarshallsSquareHeisenbergNNAFM
        self.circuit_params_dict = {'Lx' : self.Ls, 'Ly' : self.Ly, 'n_lm_neighbors' : 3}

        self.optimizer = optimizers.OptimizerGradientFree
        self.algorithm = differential_evolution
        self.opt_params_dict = {}

        return