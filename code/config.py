import numpy as np
import circuits
import hamiltonians
import optimizers
from scipy.optimize import differential_evolution, minimize
import projector
import utils

class opt_parameters:
    def __init__(self):
        self.Lx, self.Ly = 4, 4
        
        self.hamiltonian = hamiltonians.HeisenbergSquareNNNBipartitePBC;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, 'Lx' : self.Lx, 'Ly': self.Ly, 'j_pm' : +1., 'j_zz' : 1., 'j2': 0.0}

        self.circuit = circuits.SU2_PBC_symmetrized
        self.circuit_params_dict = {'Lx' : self.Lx, 'Ly' : self.Ly}


        self.projector = projector.ProjectorFull
        self.proj_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                 'generators' : [utils.get_x_symmetry_map(self.Lx, self.Ly),\
                                                 utils.get_y_symmetry_map(self.Lx, self.Ly)] ,\
                                 'eigenvalues' : [1, 1], \
                                 'degrees' : [self.Lx, self.Ly]}

        self.optimizer = optimizers.Optimizer
        self.algorithm = minimize#optimizers.natural_gradiend_descend
        self.opt_params_dict = {'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}


        return
