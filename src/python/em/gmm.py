import numpy as np

from optimization.optimizers import GradientOptimizer as GO
from optimization.stepsize import FixedScheduler as FXS

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
        self.s_current = np.zeros(self.K * 3)
        self.num_rounds = 0

    def get_update(self, sample, r_scale):

        self.num_rounds += 1

        self._compute_E_step(data, r_scale)
        self._compute_M_step()
        
    def _compute_E_step(self, data, r_scale):

        # Get weights for expected sufficient stats
        normed_data = data + np.array(
            [r_scale, -r_scale])
        densities = self._get_densities(normed_data, r_scale)
        numers = densities * self.ps
        conditional_ps = numers / np.sum(numers)

        # Get natural parameter/dual space stuff
        # (Expected sufficient stats conditioned on observations)
        s_bar = np.vstack([
            conditional_ps,
            conditional_ps * normed_data,
            conditional_ps * np.power(normed_data, 2)])

        # Do additional proximal stuff
        # (ASCENT on log-likelihood -> negate search direction)
        eta = self.eta_scheduler.get_stepsize()
        search_direction = - (s_bar - self.s_current)
        self.s_current = self.optimizer.get_update(
            self.s_current, search_direction, eta)

    def _compute_M_step(self):

        # Extract each natural parameter estimate for all components
        s_0 = self.s_current[:self.K]
        s_1 = self.s_current[self.K:2*self.K]
        s_2 = self.s_current[2*self.K:]

        self.ps = s_0

        # Calculate raw M step
        mus = s_1 / s_2
        sigmas = (s_2 - s_1) / s_0

        # Normalize for baseline effect
        self.mus = mus - self.baseline_mu
        self.sigmas = sigmas - self.baseline_sigma

    def _get_densities(self, data):

        numer = - np.power(self.mus - data, 2)
        denom = 2 * np.power(self.sigmas, 2)
        kernel = np.exp(numer / denom)
        normalizer = np.power(
            self.sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer