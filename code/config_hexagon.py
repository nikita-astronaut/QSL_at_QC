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
        self.path_to_logs = '/home/astronaut/Documents/QSL_at_QC/logs/hexagon/{:.3f}/'.format(j2)
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.mode = 'fresh'
        self.Lx, self.Ly = None, None

        ### setting up geometry and parameters ###
        self.su2 = True
        self.BC = 'PBC'
        self.spin = 0
        self.basis = ls.SpinBasis(ls.Group([]), number_spins=6, hamming_weight=3 + self.spin if self.su2 else None)
        self.basis.build()
        
        ### setting up symmetries ###
        self.symmetries = [
            utils.get_hexagon_rot_symmetry_map(basis=self.basis, su2=self.su2), \
            utils.get_hexagon_mir_symmetry_map(basis=self.basis, su2=self.su2), \
        ]
        self.eigenvalues = []#[-1, 1]
        self.sectors = []#[3, 0]  # in Toms notation
        self.degrees = []#[6, 2]

        self.unitary_no = np.ones((6, 6))

        self.unitary = self.unitary_no

        self.hamiltonian = hamiltonians.HeisenbergHexagon;
        self.ham_params_dict = {'n_qubits' : 6, \
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


        self.dimerization = [(1, 2), (3, 4), (0, 5)]
        self.circuit = circuits.SU2_symmetrized_hexagon
        self.circuit_params_dict = {'Lx' : self.Lx, \
                                    'Ly' : self.Ly, \
                                    'spin' : self.spin, \
                                    'basis' : self.basis, \
                                    'config' : self, \
                                    'unitary' : self.unitary, \
                                    'BC' : self.BC}
        

        self.projector = projector.ProjectorFull

        self.proj_params_dict = {'n_qubits' : 6, \
                                 'su2' : self.su2, \
                                 'basis' : self.basis, \
                                 'generators' : self.symmetries, \
                                 'eigenvalues' : self.eigenvalues, \
                                 'degrees' : self.degrees}

        self.observables = [observables.neel_order_hexagon(self.basis, self.su2), \
                            observables.plaquette_order_hexagon(self.basis, self.su2, self.BC) \
                           ]


        self.optimizer = optimizers.Optimizer
        self.algorithm = optimizers.natural_gradiend_descend
        self.opt_params_dict = {'lr' : 3e-3}#{'method' : 'BFGS', 'options' : {'gtol' : 1e-12, 'disp' : True}}



        #### stochastic parameters ####
        self.N_samples = None#2 ** 22
        self.SR_eig_cut = 1e-5#1e-3#3e-2#1e-2
        self.SR_diag_reg = 1e-5#1e-3#3e-2#1e-2

        return
