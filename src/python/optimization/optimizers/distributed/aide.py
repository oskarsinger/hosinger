import numpy as np
import optimization.utils as ou

from optimization.optimizers.distributed import QuasinewtonInexactDANE as QIDANE
from optimization.stepsize import FixedScheduler as FXS
from drrobert.arithmetic import get_uni_quad_sols as get_uqs
from random import choice

class AIDE:

    def __init__(self,
        model,
        servers,
        tau=0.1,
        gamma=0.8,
        mu=100,
        max_rounds=5,
        dane_rounds=50,
        init_params=None):

        self.model = model
        self.servers = servers
        self.max_rounds = max_rounds
        self.dane_rounds = dane_rounds
        # TODO: Figure out how to set self.lam (lambda) correctly.
        self.lam = 10 * np.abs(np.random.randn()) 
        self.tau = tau
        self.q = self.lam / (self.lam + self.tau)
        self.gamma = gamma
        self.mu = mu
        self.init_params = init_params
        self.w = None
        self.objectives = []

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.w)

    def compute_parameters(self):

        w_t = np.copy(self.init_params)
        y_t = np.copy(self.init_params)
        zeta_prev = np.random.uniform()

        for t in xrange(self.max_rounds):
            get_gradient = self._get_aide_gradient(y_t)
            dane_t = QIDANE(
                self.servers, 
                get_gradient,
                self.model.get_objective,
                num_rounds=self.dane_rounds,
                init_params=w_t,
                mu=self.mu)

            dane_t.compute_parameters()

            w_prev = np.copy(w_t)
            w_t = dane_t.get_parameters()
            zeta_t = self._get_zeta(zeta_prev)
            beta_t = zeta_prev * (1 - zeta_prev) / \
                (zeta_prev**2 - zeta_t)
            y_t = w_t + beta_t * (w_t - w_prev)
            zeta_prev = zeta_t
            
            self.objectives.append(
                dane_t.objectives[-1])

        self.w = w_t

    def _get_aide_gradient(self, y_t):

        def get_gradient(data, w):

            aux_term = self.tau * (w - y_t)
            original = self.model.get_gradient(data, w)

            return original + aux_term

        return get_gradient

    def _get_zeta(self, zeta_prev):

        a = 1
        b = zeta_prev**2 - self.q
        c = -zeta_prev**2
        (sol_pos, sol_neg) = get_uqs(a, b, c)
        sol = None

        # ozoi = open zero-one interval
        in_ozoi = lambda x: x > 0 and x < 1

        if in_ozoi(sol_pos) and in_ozoi(sol_neg):
            sol = choice([sol_pos, sol_neg])
        elif in_ozoi(sol_pos):
            sol = sol_pos
        else:
            sol = sol_neg

        return sol
