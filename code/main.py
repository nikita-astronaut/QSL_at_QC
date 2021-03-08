import numpy as np
import utils
import sys
import config as cv_module


config_file = utils.import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.opt_parameters()
opt_config.__dict__ = config_import.__dict__.copy()

H = opt_config.hamiltonian(**opt_config.ham_params_dict)

circuit = opt_config.circuit(**opt_config.circuit_params_dict)

opt = opt_config.optimizer(H, circuit, opt_config.algorithm, opt_config, opt_config.opt_params_dict)

print(opt.optimize())