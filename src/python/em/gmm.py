import numpy as np

from optimization.optimizers import GradientOptimizer as GO
from optimization.stepsize import FixedScheduler as FXS

# TODO: account for baseline mu and sigma in E and M steps 
class OnlineUnivariateRademacherGaussianMixtureModel:

    def __init__(self, 
        baseline_mu=0,
        baseline_sigma=0,
        optimizer=None, 
        eta_scheduler=None):

        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.K = 2

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

    def get_update(self, sample, r_scale):

        self.num_rounds += 1

        self._compute_E_step(data, r_scale)
        self._compute_M_step()
        
    def _get_E_step(self, data, r_scale):

        densities = self._get_densities(data)
        numers = densities * self.ps
        conditional_ps = numers / np.sum(numers)
        s_bar = None
        s_current = None
        eta = self.eta_scheduler.get_stepsize()
        search_direction = s_bar - s_current
        # TODO: Make sure negating search_direction is equivalent to gradient ascent
        s_new = self.optimizer.get_update(
            s_current, -search_direction, eta)

    def _get_densities(self, data):

        numer = - np.power(self.mus - data, 2)
        denom = 2 * np.power(self.sigmas, 2)
        kernel = np.exp(numer / denom)
        normalizer = np.power(
            self.sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer
