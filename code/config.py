import numpy as np
import circuits
import hamiltonians
import optimizers
import observables
from scipy.optimize import differential_evolution, minimize
import projector
import utils
import lattice_symmetries as ls
import sys
import os

### setting the MPI up ###
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()



class opt_parameters:
    def __init__(self):
        ### preparing the logging ###
        self.path_to_logs = '/home/astronaut/Documents/QSL_at_QC/logs/j2_scan_PBC_S1_11/{:.3f}/'.format(0.0)#0.1 * rank)
        os.makedirs(self.path_to_logs, exist_ok=True)


        ### setting up geometry and parameters ###
        self.Lx, self.Ly = 4, 4
        self.su2 = True
        self.BC = 'PBC'
        self.spin = 1
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=self.Lx * self.Ly, hamming_weight=self.Lx * self.Ly // 2 + self.spin if self.su2 else None)
        self.basis.build()

        self.hamiltonian = hamiltonians.HeisenbergSquare;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                'su2' : self.su2, \
                                'basis' : self.basis, \
                                'Lx' : self.Lx, 'Ly': self.Ly, \
                                'j_pm' : +1., 'j_zz' : 1., \
                                'j2': 0.0,\
                                #0.1 * rank, \
                                'BC' : self.BC}

        self.circuit = circuits.SU2_OBC_symmetrized if self.BC == 'OBC' else circuits.SU2_PBC_symmetrized
        self.circuit_params_dict = {'Lx' : self.Lx, 'Ly' : self.Ly, 'spin' : self.spin, 'basis' : self.basis}
        

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
                                 'su2' : self.su2, \
                                 'basis' : self.basis, \
                                 'generators' : [#utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
                                                 #utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
                                                 utils.get_x_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
                                                 utils.get_y_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
                                                ] ,\
                                 'eigenvalues' : [-1, -1], \
                                 'degrees' : [4, 4]}

        self.observables = [observables.neel_order(self.Lx, self.Ly, self.basis, self.su2), \
                            observables.stripe_order(self.Lx, self.Ly, self.basis, self.su2), \
                            observables.dimer_order(self.Lx, self.Ly, self.basis, self.su2, self.BC) \
                           ]


        self.optimizer = optimizers.Optimizer
        self.algorithm = optimizers.natural_gradiend_descend
        self.opt_params_dict = {'lr' : 3e-3}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}





        #### stochastic parameters ####
        self.N_samples = 2 ** 8
        self.SR_eig_cut = 1e-2
        self.SR_diag_reg = 1e-2

        return
