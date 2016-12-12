import numpy as np

from optimization.optimizers.distributed import FSVRG
from data.loaders.shortcuts import get_LRGL
from data.servers.batch import BatchServer as BS
from models import LinearRegression as LR

class GaussianLinearRegressionFSVRGRunner:

    def __init__(self,
        num_nodes,
        n,
        p,
        max_rounds=10,
        h=0.01,
        noisy=False,
        bias=False):

        self.num_nodes = num_nodes
        self.n = n
        self.max_rounds = max_rounds
        self.h = h
        self.noisy = noisy
        self.bias = bias
        self.p = p + 1 if self.bias else p

        self.init_params = np.random.randn(
            self.p * self.num_nodes, 1)

        self.w = np.random.randn(
            self.p * self.num_nodes, 1)
        ps = [p] * self.num_nodes
        ws = [np.copy(self.w[i*self.p:(i+1)*self.p])
              for i in xrange(self.num_nodes)]
        loaders = get_LRGL(
            self.n, 
            ps,
            ws=ws,
            noisys=[self.noisy] * self.num_nodes,
            bias=bias)

        self.servers = [BS(l) for l in loaders]
        self.get_model = lambda i: LR(self.p * self.num_nodes, i)
        self.w_hat = None

    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not yet been computed.')

        return self.w_hat

    def run(self):

        fsvrg = FSVRG(
            self.get_model,
            self.servers,
            max_rounds=self.max_rounds,
            init_params=self.init_params,
            h=self.h)

        fsvrg.compute_parameters()

        self.w_hat = fsvrg.get_parameters()
        self.objectives = fsvrg.objectives
