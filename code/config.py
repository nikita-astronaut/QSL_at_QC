import numpy as np
import circuits
import hamiltonians
import optimizers
from scipy.optimize import differential_evolution, minimize


class opt_parameters:
    def __init__(self):
        self.Lx, self.Ly = 2, 4
        
        self.hamiltonian = hamiltonians.HeisenbergSquareNNBipartiteSparseOBC;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, 'Lx' : self.Lx, 'Ly': self.Ly, 'j_pm' : -1., 'j_zz' : 1.}

        self.circuit = circuits.TrotterizedMarshallsSquareHeisenbergNNAFM
        self.circuit_params_dict = {'Lx' : self.Lx, 'Ly' : self.Ly, 'n_lm_neighbors' : 3}

        self.optimizer = optimizers.Optimizer
        self.algorithm = minimize
        self.opt_params_dict = {'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}


        return
