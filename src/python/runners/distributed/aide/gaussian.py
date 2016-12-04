import numpy as np

from optimization.optimizers.distributed import AIDE
from data.loaders.synthetic.shortcuts import get_LRGL
from data.servers.batch import BatchServer as BS
from models import GaussianLinearRegression as GLR

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
        self.model = GLR(self.p) 
        self.w_hat = None

    def get_parameters(self):

        return self.w_hat

    def run(self):
        
        aide = AIDE(
            self.model,
            self.servers,
            max_rounds=self.max_rounds,
            tau=self.tau,
            gamma=self.gamma,
            dane_rounds=self.dane_rounds,
            init_params=self.init_params)

        aide.compute_parameters()

        self.w_hat = aide.get_parameters()
        self.errors = aide.errors
