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
        # n_trial = int(sys.argv[3])
        ### preparing the logging ###
        self.path_to_logs = '/home/astronaut/Documents/QSL_at_QC/logs/1l/{:.3f}/'.format(j2)
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.mode = 'continue'
        #self.start_params = np.array([-0.1829, -0.1500, -0.0237, 0.0310, 0.0318, 0.0596, 0.0199, 0.0376, -0.0676, -0.0636, 0.0466, 0.1005, -0.0078, 0.0068, 0.0091, 0.0050, 0.0587, 0.0307, 0.0567, 0.0376, 0.0538, 0.0110, -0.0175, 0.0552, -0.1326, -0.0185, -0.0123, -0.1392, -0.0766, -0.0766, -0.0808, -0.0775, 0.2917, 0.2877, -0.2865, -0.2896, -0.2872, -0.2722, -0.2835, -0.2828, 0.0344, 0.0151, 0.0196, 0.0359, -0.0801, -0.0836, -0.0830, -0.0788, -0.3887, -0.5589, -0.3637, -0.5764, -0.3460, -0.5901, -0.2157, 1.1598, -0.0107, -0.0085, -0.0113, -0.0083, 0.0425, 0.0371, 0.0048, 0.0043])

        self.target_norm = 0.20 #0.20#0.10#0.20 #0.80
        self.lagrange = True
        self.Z = 300.

        self.test = False
        self.reg = 'diag'

        ### setting up geometry and parameters ###
        self.Lx, self.Ly, self.subl = 4, 4, 1
        self.su2 = False#True
        self.BC = 'PBC'
        self.spin = 0
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=self.Lx * self.Ly, hamming_weight=self.Lx * self.Ly // 2 + self.spin if self.su2 else None)
        self.basis.build()
        
        ### setting up symmetries ###
        self.symmetries = [
            utils.get_x_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_y_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_rot_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            utils.get_Cx_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2), \
            #utils.get_Cy_symmetry_map(self.Lx, self.Ly, basis=self.basis, su2=self.su2)
            
        ]
        self.eigenvalues = [1, 1, 1, 1]#1, -1, 1]#, 1]
        self.sectors = [0, 0, 0, 0]#[0, 2, 0]#, 0]  # in Toms notation
        self.degrees = [4, 4, 4, 2]#[4, 4, 2]#, 2]

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
                                'permutations' : [s[1] for s in self.symmetries], \
                                'sectors' : self.sectors, \
                                'spin' : self.spin, \
                                'unitary' : self.unitary
                                }


        self.dimerization = [(0, 5), (1, 4), (2, 7), (3, 6), (8, 13), (9, 12), (10, 15), (11, 14)] if j2 > 0.7 else [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        self.circuit = circuits.SU2_symmetrized
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
        self.algorithm = optimizers.natural_gradiend_descend
        self.opt_params_dict = {'lr' : 3e-4}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}



        #### stochastic parameters ####
        self.N_samples = 2 ** 16 #2 ** 12
        self.SR_eig_cut = 1e-2#3e-2#1e-2
        self.SR_diag_reg = 0.#1e-4#3e-2#1e-2


        #### noise parameters ####
        self.qiskit = True
        if self.qiskit:
            import qiskit.providers.aer.noise as noise
            self.prob_1 = 1e-4
            self.prob_2 = 1e-3
            error_1 = noise.depolarizing_error(self.prob_1, 1)
            error_2 = noise.depolarizing_error(self.prob_2, 2)

            # Add errors to noise model
            self.noise_model = noise.NoiseModel()
            #self.noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            self.noise_model.add_all_qubit_quantum_error(error_2, ['swap', 'eswap'])#['cx'])




        return


