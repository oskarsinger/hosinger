import numpy as np

from optimization.optimizers import FederatedOptimizer as FDO
from optimization.stepsize import FixedScheduler as FXS

class NetworkInterferenceLearner:

    def __init__(self, 
        id_num, 
        adj_matrix, 
        burn_in,
        phi=lambda x: x**(0.5)):

        self.id_num = self.id_num
        self.adj_matrix = adj_matrix
        self.burn_in = burn_in
        self.phi = phi

        self.num_nodes = self.adj_matrix.shape[0]
        self.mus = np.random.randn(self.num_nodes, 2)
        self.sigmas = np.abs(np.random.randn(self.num_nodes, 2))
        self.ps = np.random.uniform(size=self.num_nodes)
        self.optimizer = FDO()
        self.eta_scheduler = FXS(0.1)
        self.num_rounds = 0
        self.action_history = []
        self.feedback_history = []

    def get_action(self):

        self.num_rounds += 1

        action = None

        if self.num_rounds <= self.burn_in:
            action = 0
        else:
            Sk = self._get_Sk()

            self.history.append(Sk)

            action = int(self.id_num in Sk)

        return action

    def set_feedback(self, f):

        self.feedback_history.append(f)
        self._update_parameters()

    def _update_parameters(self):

        f = self.feedback_history[-1]

        self._compute_E_step()
        self._compute_M_step()

    def _get_E_step(self):

        eta = self.eta_scheduler.get_stepsize()
        E_step = None

        self.ps = self.optimizer.get_update(
            self.ps, E_step, eta)

    def _get_M_step(self):

        self.mus = 'Poop'
