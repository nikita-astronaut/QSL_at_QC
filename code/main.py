import numpy as np
import utils
import sys
import config as cv_module
import lattice_symmetries as ls
import scipy.linalg
import observables

#### TEST #####
'''
sx = np.array(
               [[0, 1], \
               [1, 0]]
              )
sy = np.array([[0, 1.0j], [-1.0j, 0]])
sz = np.array([[1, 0], \
               [0, -1]])

SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
print((SS + np.eye(4)) / 2.)
matrix = scipy.linalg.expm(1.0j * 0.2 * SS)
basis = ls.SpinBasis(ls.Group([]), number_spins=2, hamming_weight=None)
basis.build()
op = ls.Operator(basis, [ls.Interaction(matrix.conj(), [(0, 1)])])

for i in range(4):
	state = np.zeros(4, dtype=np.complex128)
	state[i] = 1.
	assert np.allclose(matrix.dot(state), op(state))
exit(-1)
'''


config_file = utils.import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.opt_parameters()
opt_config.__dict__ = config_import.__dict__.copy()

H = opt_config.hamiltonian(**opt_config.ham_params_dict)

circuit = opt_config.circuit(**opt_config.circuit_params_dict)

projector = opt_config.projector(**opt_config.proj_params_dict)
obs = observables.Observables(opt_config, H, circuit, projector)

opt = opt_config.optimizer(H, circuit, projector, obs, opt_config.algorithm, opt_config, opt_config.opt_params_dict)


print(opt.optimize())
