import numpy as np
import utils
import sys
import config as cv_module

config_file = import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.MC_parameters()
opt_config.__dict__ = config_import.__dict__.copy()

H = opt_config.hamiltonian(**opt_config.ham_params_dict)
circuit = opt_config.circuit(**opt_config.circuit_params_dict)
opt = opt_config.circuit(H, circuit, **opt_config.opt_params_dict)


print(opt.optimize())