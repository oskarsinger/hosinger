import numpy as np

from optimization.optimizers.distributed import FSVRG
from data.loaders.shortcuts import get_LRGL
from data.servers.batch import BatchServer as BS
from models import LinearRegression as LR

class GaussianLinearRegressionSVRGRunner:

    def __init__(self,
        num_nodes,
        n,
        p,
        max_rounds=5,
        h=0.01,
        init_params=None,
        noisy=False):

        self.num_nodes = num_nodes
        self.n = n
        # Add 1 for bias term
        self.p = p + 1
        self.max_rounds = max_rounds
        self.h = h
        self.noisy = noisy

        if init_params is None:
            init_params = np.random.randn(self.p, 1)

        self.init_params = init_params

        self.w = np.random.randn(self.p, 1)
        ps = [p] * self.num_nodes
        ws = [np.copy(self.w) 
              for i in xrange(self.num_nodes)]
        loaders = get_LRGL(
            self.n, 
            ps,
            ws = ws,
            noisys=[self.noisy] * self.num_nodes,
            bias=True)

        self.servers = [BS(l) for l in loaders]
        self.model = LR(self.p) 
        self.w_hat = None

    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not yet been computed.')

        return self.w_hat

    def run(self):

        fsvrg = FSVRG(
            self.model,
            self.servers,
            max_rounds=self.max_rounds,
            init_params=self.init_params,
            h=self.h)

        fsvrg.compute_parameters()

        self.w_hat = fsvrg.get_parameters()
        self.errors = fsvrg.errors
