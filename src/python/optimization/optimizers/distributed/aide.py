import numpy as np
import optimization.utils as ou
import QuasinewtonInexactDANE as QIDANE

from optimization.stepsize import FixedScheduler as FXS

class AIDE:

    def __init__(self,
        servers,
        get_gradient,
        max_rounds=5,
        dane_rounds=3,
        init_params=None):

        self.servers = servers
        self.get_gradient = get_gradient
        self.max_rounds = max_rounds
        self.dane_rounds = dane_rounds
        self.gamma = gamma
        self.tau = tau
        self.init_params = init_params
        self.w = None

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
                init_params=w_t)

            dane_t.compute_parameters()

            w_prev = np.copy(w_t)
            w_t = dane_t.get_parameters()
            zeta_t = self._get_zeta(zeta_prev)
            beta_t = zeta_prev * (1 - zeta_prev) / \
                (zeta_prev**2 - zeta_t)
            y_t = w_t + beta_t * (w_t - w_prev)
            zeta_prev = zeta_t

        self.w = w_t

    def _get_aide_gradient(self, y_t):

        def get_gradient(data, w):

            aux_term = self.tau * (w - y_t)
            original = self.get_gradient(data, w)

            return original + aux_term

        return get_gradient

    def _get_zeta(self, zeta_prev):

        print 'Poop'
