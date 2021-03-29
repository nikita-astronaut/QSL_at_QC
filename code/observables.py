import numpy as np
import lattice_symmetries as ls
import os



sz = np.array([[1, 0], \
               [0, -1]])

sx = np.array(
               [[0, 1], \
               [1, 0]]
              )

sy = np.array([[0, 1.0j], [-1.0j, 0]])

SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)

def neel_order(Lx, Ly, basis, su2=False):
    n_qubits = Lx * Ly

    site_plus = []
    site_minus = []
    for x in range(Lx):
        for y in range(Ly):
            if x + y % 2 == 0:
                site_plus.append(x + y * Lx)
            else:
                site_minus.append(x + y * Lx)

    return ls.Operator(basis, [ls.Interaction(sz / n_qubits, site_plus), ls.Interaction(-sz / n_qubits, site_minus)]), 'Neel'

def stripe_order(Lx, Ly, basis, su2=False):
    n_qubits = Lx * Ly

    site_plus = []
    site_minus = []
    for x in range(Lx):
        for y in range(Ly):
            if x % 2 == 0:
                site_plus.append(x + y * Lx)
            else:
                site_minus.append(x + y * Lx)

    return ls.Operator(basis, [ls.Interaction(sz / n_qubits, site_plus), ls.Interaction(-sz / n_qubits, site_minus)]), 'Stripe'


def dimer_order(Lx, Ly, basis, su2=False, BC='PBC'):
    n_qubits = Lx * Ly
    

    bond_plus = []
    bond_minus = []
    for x in range(Lx):
        for y in range(Ly):
            if x % 2 == 0:
                if x < Lx - 1 or BC == 'PBC':
                    bond_plus.append(((x + y * Lx), (((x + 1) % Lx) + y * Lx)))
            else:
                if x < Lx - 1 or BC == 'PBC':
                    bond_minus.append(((x + y * Lx), (((x + 1) % Lx) + y * Lx)))

    return ls.Operator(basis, [ls.Interaction(SS / n_qubits, bond_plus), ls.Interaction(-SS / n_qubits, bond_minus)]), 'Dimer'


class Observables(object):
    def __init__(self, config, hamiltonian, circuit, projector):
        self.path_to_logs = config.path_to_logs

        self.main_log = open(os.path.join(self.path_to_logs, 'main_log.dat'), 'w')
        self.force_log = open(os.path.join(self.path_to_logs, 'force_log.dat'), 'w')
        self.exact_force_log = open(os.path.join(self.path_to_logs, 'exact_force_log.dat'), 'w')
        self.force_SR_log = open(os.path.join(self.path_to_logs, 'force_SR_log.dat'), 'w')
        self.exact_force_SR_log = open(os.path.join(self.path_to_logs, 'exact_force_SR_log.dat'), 'w')

        self.parameters_log = open(os.path.join(self.path_to_logs, 'parameters_log.dat'), 'w')

        self.observables = config.observables
        self.hamiltonian = hamiltonian
        self.projector = projector
        self.circuit = circuit


        ### prepare main log ###
        string = 'gsenergy energy fidelity '
        for _, name in self.observables:
            string += name + ' '
        self.main_log.write(string + '\n')

        return

    def write_logs(self):
        force_exact = self.circuit.forces_exact
        for f in force_exact:
            self.exact_force_log.write('{:.4f} '.format(f))
        self.exact_force_log.write('\n')
        self.exact_force_log.flush()


        force = self.circuit.forces
        for f in force:
            self.force_log.write('{:.4f} '.format(f))
        self.force_log.write('\n')
        self.force_log.flush()

        force_SR_exact = self.circuit.forces_SR_exact
        for f in force_SR_exact:
            self.exact_force_SR_log.write('{:.4f} '.format(f))
        self.exact_force_SR_log.write('\n')
        self.exact_force_SR_log.flush()


        force_SR = self.circuit.forces_SR
        for f in force_SR:
            self.force_SR_log.write('{:.4f} '.format(f))
        self.force_SR_log.write('\n')
        self.force_SR_log.flush()


        parameters = self.circuit.params
        for p in parameters:
            self.parameters_log.write('{:.4f}, '.format(p))
        self.parameters_log.write('\n')
        self.parameters_log.flush()


        #### compute energy ###
        state = self.circuit()
        state_proj = self.projector(state)
        norm = np.dot(state.conj(), state_proj)
        energy = (np.dot(np.conj(state), self.hamiltonian(state_proj)) / norm).real - self.hamiltonian.energy_renorm
        

        ### compute fidelity ###
        state_proj = state_proj / np.sqrt(norm)
        assert np.isclose(np.dot(state_proj, state_proj.conj()), 1.0)
        fidelity = np.abs(np.dot(self.hamiltonian.ground_state[0].conj(), state_proj)) ** 2

        obs_vals = []
        for operator, _ in self.observables:
            val = np.dot(state_proj.conj(), operator(operator(state_proj)))
            assert np.isclose(val.imag, 0.0)
            obs_vals.append(val.real)
        
        #### dimer amendment ###
        dimer_avg = np.dot(state_proj.conj(), self.observables[-1][0](state_proj))
        assert np.isclose(dimer_avg.imag, 0.0)
        obs_vals[-1] -= (dimer_avg ** 2).real

        self.main_log.write(('{:.7f} {:.7f} {:.7f} ' + '{:.7f} ' * len(obs_vals) + '\n').format(self.hamiltonian.gse - self.hamiltonian.energy_renorm, energy, fidelity, *obs_vals))
        self.main_log.flush()
