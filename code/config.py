import numpy as np
import circuits
import hamiltonians
import optimizers
from scipy.optimize import differential_evolution, minimize
import projector
import utils


class opt_parameters:
    def __init__(self):
        self.path_to_logs = '/home/astronaut/QSL_at_QC/logs/'
        self.Lx, self.Ly = 4, 4
        su2 = True

        self.hamiltonian = hamiltonians.HeisenbergSquare;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                'su2' : su2, \
                                'Lx' : self.Lx, 'Ly': self.Ly, \
                                'j_pm' : +1., 'j_zz' : 1., \
                                'j2': 0.5,
                                'BC' : 'OBC'}

        self.circuit = circuits.SU2_OBC_symmetrized
        self.circuit_params_dict = {'Lx' : self.Lx, 'Ly' : self.Ly}
        

        self.projector = projector.ProjectorFull
        '''
        self.proj_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                 'su2' : True, \
                                 'generators' : [utils.get_x_symmetry_map(self.Lx, self.Ly, su2=su2),\
                                                 utils.get_y_symmetry_map(self.Lx, self.Ly, su2=su2), \
                                                 utils.get_rot_symmetry_map(self.Lx, self.Ly, su2=su2), \
                                                 utils.get_Cx_symmetry_map(self.Lx, self.Ly, su2=su2)] ,\
                                 'eigenvalues' : [1, 1, 1, 1], \
                                 'degrees' : [self.Lx, self.Ly, 4, 2]}
        '''

        self.proj_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                 'su2' : su2, \
                                 'generators' : [utils.get_rot_symmetry_map(self.Lx, self.Ly, su2=su2), \
                                                 utils.get_Cx_symmetry_map(self.Lx, self.Ly, su2=su2)] ,\
                                 'eigenvalues' : [1, 1], \
                                 'degrees' : [4, 2]}


        self.optimizer = optimizers.Optimizer
        self.algorithm = optimizers.natural_gradiend_descend
        self.opt_params_dict = {}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}





        #### stochastic parameters ####
        self.N_samples = 2 ** 13
        self.SR_eig_cut = 1e-2
        self.SR_diag_reg = 1e-2

        return
