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
        self.with_mpi = True
        j2 = float(sys.argv[2])
        # n_trial = int(sys.argv[3])
        ### preparing the logging ###
        self.path_to_logs = '/home/astronaut/Documents/QSL_at_QC/logs/tests2_2layers/{:.3f}/'.format(j2)
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.mode = 'continue'#preassigned'
        #self.start_params = np.array([-0.3685, 0.4482, -0.0392, 0.2020, 0.0176, 0.0232, 0.2243, -0.1977, -0.0397, 0.1452, 0.0977, 0.1164, -0.0738, -0.0676, 0.2746, -0.2067, -0.0607, 0.1969, -0.1255, -0.0639, 0.0043, 0.1844, 0.1209, -0.2012, -0.7399, 0.0994, -0.0190, -0.0755, 0.0712, 0.0217, 0.0836, 0.0750, 0.5846, 0.0583, -0.0376, 0.8201, 0.3595, 0.2171, 0.1347, 0.4565, 0.2980, 0.1066, 0.1914, -0.2382, 0.0938, 0.0925, 0.0987, 0.0874, 0.3376, 0.2784, 0.5012, 0.4006, 0.5574, 0.3548, 0.2837, 0.3671, -0.4296, 0.0146, 0.0455, 0.0115, -0.0268, -0.0026, 0.0025, -0.0297,])
        
        self.target_norm = 0.25 #0.5#0.5 #0.20#0.10#0.20 #0.80
        self.lagrange = False#True #True #False#True# True
        self.Z = 300.

        self.test = False # False#True#False# True#False#True#False
        self.reg = 'diag'

        ### setting up geometry and parameters ###
        self.Lx, self.Ly, self.subl = 5, 4, 1
        self.su2 = True#False#True
        self.BC = 'PBC'
        self.spin = 0
        self.noise = False; assert not (self.noise and self.su2)
        self.noise_p = 0.#1e-2#3e-3 #3e-3#3e-3


        self.basis = ls.SpinBasis(ls.Group([]), number_spins=self.Lx * self.Ly, hamming_weight=self.Lx * self.Ly // 2 + self.spin if self.su2 else None)
        self.basis.build()
        
        ### setting up symmetries ###
        self.symmetries = [
            #utils.get_x_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_y_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
            #utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_Cy_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
        ]
        self.eigenvalues = [1, 1, 1]#, 1]#1, -1, 1]#, 1]
        self.sectors = [0, 0, 0]#, 0]#[0, 2, 0]#, 0]  # in Toms notation
        self.degrees = [4, 2, 2]#[4, 4, 2]#, 2]

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


        self.hamiltonian = hamiltonians.HeisenbergSquare_5x4;
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                'su2' : self.su2, \
                                'basis' : self.basis, \
                                'Lx' : self.Lx, 'Ly': self.Ly, \
                                'j_pm' : +1., 'j_zz' : 1., \
                                'j2': j2, \
                                'BC' : self.BC, \
                                'symmetries' : [s[0] for s in self.symmetries], \
                                'permutations' : [s[1] for s in self.symmetries], \
                                'sectors' : self.sectors, \
                                'spin' : self.spin, \
                                'unitary' : self.unitary, \
                                'workdir' : self.path_to_logs
                                }


        #self.dimerization = [(0, 5), (1, 4), (2, 7), (3, 6), (8, 13), (9, 12), (10, 15), (11, 14)] if j2 > 0.7 else [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        #self.dimerization = [(0, 1), (2, 3), (4, 5), (6, 7)]#, (8, 9), (10, 11), (12, 13)]
        #self.dimerization = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23)]
        #self.dimerization = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        self.dimerization = [(0, 5), (10, 15), (1, 6), (11, 16), (2, 7), (12, 17), (3, 8), (13, 18), (4, 9), (14, 19)]
        #self.dimerization = [(0, 1), (2, 3), (5, 6), (7, 8), (10, 11), (12, 13), (15, 16), (17, 18), (4, 9), (14, 19)]

        #self.dimerization = [(0, 6), (1, 5), (2, 8), (3, 7), (4, 9), (14, 19), (10, 16), (11, 15), (12, 18), (13, 17)]
        self.circuit = circuits.SU2_symmetrized_square_5x4_OBCPBC
        self.circuit_params_dict = {'Lx' : self.Lx, \
                                    'Ly' : self.Ly, \
                                    'subl' : self.subl, \
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
        self.algorithm = optimizers.Lanczos_energy_extrapolation #natural_gradiend_descend#SPSA_gradiend_descend# projected_energy_estimation #optimizers.SPSA_gradiend_descend
        self.write_logs = True

        self.opt_params_dict = {}#'lr' : 1e-3}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}
        self.SPSA_epsilon = 3e-2; self.max_energy_increase_threshold = None; self.SPSA_hessian_averages = 1; self.SPSA_gradient_averages = 1



        #### stochastic parameters ####
        self.N_samples = 2 ** int(sys.argv[3]) #int(sys.argv[3])#2 ** int(sys.argv[3])
        self.SR_eig_cut = 3e-2
        self.SR_diag_reg = 0.


        ### Lanczos parameters ###
        self.max_Lanczos_order = 8


        #### noise parameters ####
        self.qiskit = False#True #True # True
        if self.qiskit:
            import qiskit.providers.aer.noise as noise
            self.prob_1 = float(sys.argv[4]) / 10. #1e-4#1e-9
            self.prob_2 = self.prob_1 * 10. ##1e-9
            error_1 = noise.depolarizing_error(self.prob_1, 1)
            error_2 = noise.depolarizing_error(self.prob_2, 2)

            # Add errors to noise model
            self.noise_model = noise.NoiseModel()
            self.noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'h'])
            self.noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            self.n_noise_repetitions = int(sys.argv[3])
            self.test_trials = 100


        return
