import numpy as np

from optimization.optimizers.distributed import AIDE
from data.loaders.synthetic.shortcuts import get_LRGL
from data.servers.batch import BatchServer as BS

class GaussianLinearRegressionAIDERunner:

    def __init__(self,
        num_nodes,
        n,
        p,
        max_rounds=5,
        dane_rounds=3,
        tau=0.1,
        gamma=0.8,
        init_params=None):

        self.num_nodes = num_nodes
        self.n = n
        self.p = p
        self.max_rounds = max_rounds
        self.dane_rounds = dane_rounds
        self.tau = tau
        self.gamma = gamma

        if init_params is None:
            init_params = np.random.randn(self.p, 1)

        self.init_params = init_params

        self.w = np.random.randn(self.p, 1)
        ps = [self.p] * self.num_nodes
        ws = [np.copy(self.w) 
              for i in xrange(self.num_nodes)]
        loaders = get_LRGL(
            self.n, 
            ps,
            ws = ws)
        self.servers = [BS(l) for l in loaders]
        self.w_hat = None

    def get_parameters(self):

        return self.w_hat

    def run(self):
        
        aide = AIDE(
            self.servers,
            self.get_gradient,
            max_rounds=self.max_rounds,
            tau=self.tau,
            gamma=self.gamma,
            dane_rounds=self.dane_rounds,
            init_params=self.init_params)

        aide.compute_parameters()

        self.w_hat = aide.get_parameters()

        print 'Error', np.linalg.norm(self.w_hat - self.w)

    def get_gradient(self, data, params):

        (A, b) = data
        (n, p) = A.shape

        if params is None:
            params = np.random.randn(p, 1)

        return np.dot(A, params) - b
