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
        j2 = float(sys.argv[2])
        ### preparing the logging ###
        self.path_to_logs = '/home/astronaut/Documents/QSL_at_QC/logs/1sample/{:.3f}/'.format(j2)
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.mode = 'continue'
        #self.start_params = np.array([-0.0320, -0.0814, 0.0971, -0.4499, -0.0189, -0.2592, 1.5065, 0.1660, -0.3358, 0.1532, -0.4515, -0.0111, 0.3332, 0.1158, -0.2977, 0.2625, 0.0813, 0.0803, -1.0671, 0.0874, 1.1376, 0.0195, -0.1013, -0.0069, -0.0807, -0.1234, 0.0501, -0.0285, 1.5076, -0.0157, 0.0737, 0.8962, -0.0001, -0.7749, -0.1028, -0.1581, 0.1752, 0.1576, -0.4152, 0.0646, 0.0203, 0.3912, 0.2760, -0.0792, -0.0063, 0.0939, 0.0928, 0.2466, 0.2254, 1.5382, -0.0062, -0.0661, -0.1958, 1.4844, -0.0668, 0.2281, -0.8117, 0.9396, 0.7315, 0.7567, -0.0486, 0.0893, -0.2749, -0.0386])

        ### setting up geometry and parameters ###
        self.Lx, self.Ly = 4, 4
        self.su2 = True
        self.BC = 'PBC'
        self.spin = 0
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=self.Lx * self.Ly, hamming_weight=self.Lx * self.Ly // 2 + self.spin if self.su2 else None)
        self.basis.build()
        

        ### setting up symmetries ###
        self.symmetries = [
            utils.get_x_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_y_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cy_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
            
        ]
        self.eigenvalues = [1, -1]
        self.sectors = [0, 2]  # in Toms notation
        self.degrees = [4, 4]

        self.unitary_no = np.ones((self.Lx * self.Ly, self.Lx * self.Ly))
        self.unitary_neel = np.ones((self.Lx * self.Ly, self.Lx * self.Ly))
        self.unitary_stripe = np.ones((self.Lx * self.Ly, self.Lx * self.Ly))

        self.unitary = self.unitary_no#neel#stripe

        for i in range(self.Lx * self.Ly):
            for j in range(self.Lx * self.Ly):
                if (i // self.Lx) % 2 != (j // self.Lx) % 2:
                    self.unitary_stripe[i, j] = -1
                if (i % self.Lx + i // self.Lx) % 2 != (j % self.Lx + j // self.Lx) % 2:
                    self.unitary_neel[i, j] = -1


        self.hamiltonian = hamiltonians.HeisenbergSquare;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                'su2' : self.su2, \
                                'basis' : self.basis, \
                                'Lx' : self.Lx, 'Ly': self.Ly, \
                                'j_pm' : +1., 'j_zz' : 1., \
                                'j2': j2, \
                                'BC' : self.BC, \
                                'symmetries' : [s[0] for s in self.symmetries], \
                                'sectors' : self.sectors, \
                                'spin' : self.spin, \
                                'unitary' : self.unitary
                                }


        self.dimerization = [(0, 5), (1, 4), (2, 7), (3, 6), (8, 13), (9, 12), (10, 15), (11, 14)] if j2 > 0.6 else [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        self.circuit = circuits.SU2_symmetrized
        self.circuit_params_dict = {'Lx' : self.Lx, \
                                    'Ly' : self.Ly, \
                                    'spin' : self.spin, \
                                    'basis' : self.basis, \
                                    'config' : self, \
                                    'unitary' : self.unitary, \
                                    'BC' : self.BC}
        

        self.projector = projector.ProjectorFull

        self.proj_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                 'su2' : self.su2, \
                                 'basis' : self.basis, \
                                 'generators' : self.symmetries, \
                                 'eigenvalues' : self.eigenvalues, \
                                 'degrees' : self.degrees}

        self.observables = [observables.neel_order(self.Lx, self.Ly, self.basis, self.su2), \
                            observables.stripe_order(self.Lx, self.Ly, self.basis, self.su2), \
                            observables.dimer_order(self.Lx, self.Ly, self.basis, self.su2, self.BC) \
                           ]


        self.optimizer = optimizers.Optimizer
        self.algorithm = optimizers.natural_gradiend_descend
        self.opt_params_dict = {'lr' : 3e-3}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}



        #### stochastic parameters ####
        self.N_samples = None#2 ** 22
        self.SR_eig_cut = 1e-3#3e-2#1e-2
        self.SR_diag_reg = 1e-3#3e-2#1e-2

        return
