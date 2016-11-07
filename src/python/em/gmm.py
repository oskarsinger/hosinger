import numpy as np

from optimization.optimizers import GradientOptimizer as GO
from optimization.stepsize import FixedScheduler as FXS

class OnlineUnivariateGaussianMixtureModel:

    def __init__(self, 
        K=2,
        optimizer=None, 
        eta_scheduler=None):

        self.K = K

        if optimizer is None:
            self.optimizer = GO()

        self.optimizer = optimizer

        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler

        ps = np.random.randint(low=0, size=self.K)
        self.ps = ps / np.sum(ps)
        self.mus = np.zeros(self.K)
        self.sigmas = np.ones(self.K)
        self.num_rounds = 0

    def get_update(self, data):

        self.num_rounds += 1

        self._compute_E_step(data)
        
    def _get_E_step(self, data):

        densities = self._get_densities(data)
        numers = densities * self.ps
        numers / np.sum(numers)

    def _get_densities(self, data):

        numer = - np.power(self.mus - data, 2)
        denom = 2 * np.power(self.sigmas, 2)
        kernel = np.exp(numer / denom)
        normalizer = np.power(
            self.sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer
