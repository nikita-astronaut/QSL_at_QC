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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



class opt_parameters:
    def __init__(self):
        self.with_mpi = True
        j2 = float(sys.argv[2])
        n_trial = int(sys.argv[3])
        ### preparing the logging ###
        #self.path_to_logs = '/home/cluster/niastr/data/QSL_at_QC//logs/qiskit_2x{:d}_experiments/{:.3f}_{:d}_{:d}_{:.5f}/'.format(j2, int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        self.path_to_logs = '/home/cluster/niastr/data/logs_SR1e-1_allsymm_depth_lagrange/TFIM_{:d}/{:.3f}_{:d}_{:d}/'.format(int(sys.argv[5]), j2, int(sys.argv[3]), int(sys.argv[4]))
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.mode = 'continue'
        
        self.target_norm = 0.98
        self.lagrange = True if int(sys.argv[6]) == 1 else False
        self.Z = 300.

        self.test = False
        self.reg = 'diag'

        ### setting up geometry and parameters ###
        self.Lx, self.Ly, self.subl = 1, int(sys.argv[5]), 1
        self.su2 = False
        self.BC = 'PBC'
        self.spin = 0
        self.noise = False; assert not (self.noise and self.su2)
        self.noise_p = 0.


        self.basis = ls.SpinBasis(ls.Group([]), number_spins=self.Lx * self.Ly, hamming_weight=(self.Lx * self.Ly) // 2 + self.spin if self.su2 else None)
        self.basis.build()
        
        ### setting up symmetries ###
        self.symmetries = [
            #utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_y_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_Cy_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
            #utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cy_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
        ]

        if int(sys.argv[6]) == 1:
            self.eigenvalues = [1, 1]
            self.sectors = [0, 0]
            self.degrees = [self.Ly, 2]
        else:
            self.eigenvalues = []
            self.sectors = []
            self.degrees = []

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


        self.hamiltonian = hamiltonians.TFIMChain
        self.ham_params_dict = {'n_qubits' : self.Lx * self.Ly, \
                                'su2' : self.su2, \
                                'basis' : self.basis, \
                                'Lx' : self.Lx, 'Ly': self.Ly, \
                                'h': j2, \
                                'xBC' : 'OBC', \
                                'yBC' : self.BC, \
                                'symmetries' : [s[0] for s in self.symmetries], \
                                'permutations' : [s[1] for s in self.symmetries], \
                                'sectors' : self.sectors, \
                                'spin' : self.spin, \
                                'unitary' : self.unitary, \
                                'workdir' : self.path_to_logs
                                }


        self.dimerization = 'AFM' if j2 < 1.0 else 'para'#[(2 * i, 2 * i + 1) for i in range(self.Ly)]
        self.circuit = circuits.TFIM_1xL
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
                            #observables.dimer_order(self.Lx, self.Ly, self.basis, self.su2, self.BC) \
                           ]


        self.optimizer = optimizers.Optimizer
        self.algorithm = optimizers.natural_gradiend_descend #SPSA_gradiend_descend#Lanczos_energy_extrapolation #natural_gradiend_descend#SPSA_gradiend_descend# projected_energy_estimation #optimizers.SPSA_gradiend_descend
        self.write_logs = True

        self.opt_params_dict = {'lr' : 3e-3}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}
        self.SPSA_epsilon = 3e-2; self.max_energy_increase_threshold = None; self.SPSA_hessian_averages = 1; self.SPSA_gradient_averages = 1



        #### stochastic parameters ####
        self.N_samples = 2 ** int(sys.argv[4]) #int(sys.argv[3])#2 ** int(sys.argv[3])
        self.SR_eig_cut = 1e-1
        self.SR_diag_reg = 0.
        self.SR_scheduler = False#True


        ### Lanczos parameters ###
        self.max_Lanczos_order = 2;


        #### noise parameters ####
        self.qiskit = False #True #False#True #True # True
        if self.qiskit:
            import qiskit.providers.aer.noise as noise
            self.prob_1 = float(sys.argv[5]) / 10. #1e-4#1e-9
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
