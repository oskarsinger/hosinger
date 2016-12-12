import numpy as np

from drrobert.misc import unzip
from scipy.stats import norm

class BanditNetworkRademacherGaussianMixtureModel:

    def __init__(self, 
        budget,
        id_number
        baseline_mu=0,
        baseline_sigma=0):

        self.budget = budget
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.K = 2
        self.p = self.K * 6
        ps = np.random.randint(low=0, size=self.K)
        self.ps = ps / np.sum(ps)
        self.mus = np.zeros(self.K)
        self.sigmas = np.ones(self.K)

    def get_action(self, params):

        max_id = int(params.shape[0] / self.p)
        Sk = np.random.choice(
            max_index,
            replace=False,
            self.budget).tolist()

        return int(self.id_number in Sk)

    def get_objective(self, data, params):

        objective = 0

        for ((reward, r_scale), action) in data:
            shifted_sample = sample + np.hstack(
                [r_scale, -r_scale])
            densities = self._get_densities(
                shifted_sample)
            likelihood = np.dot(densities, self.ps)

            objective -= np.log(likelihood)

        return objective

    def get_residuals(self, data, params):
        
        raise Exception(
            'This method is not implemented for this class.')

    def get_coordinate_counts(self, data):

        actions = unzip(data)[1]
        total = len(actions)
        acted = sum(actions)

        coord_counts = np.zeros(
            self.get_parameter_shape())

        coord_counts[0:2,:] = total
        coord_counts[2:,:] = acted

        return coord_counts

    def get_parameter_shape(self):

        return (self.p, 1)

    def get_gradient(self, data, params):

        ((sample, r_scale), action) = data
        gradient = self._get_gradient(
            sample, r_scale, action)

        self._compute_M_step(params, action)

        return gradient
        
    def _get_gradient(self, sample, r_scale, action):

        # Get weights for expected sufficient stats
        shifted_sample = sample + np.hstack(
            [r_scale, -r_scale])
        densities = self._get_densities(
            shifted_sample, r_scale)
        numers = densities * self.ps
        conditional_ps = numers / np.sum(numers)

        # TODO: make sure this is the correct way to update when no treatment is applied
        if action == 0:
            shifted_sample = np.zeros_like(
                shifted_sample)

        # Get natural parameter/dual space stuff
        # (Expected sufficient stats conditioned on observations)
        s_bar = np.vstack([
            conditional_ps,
            conditional_ps * \
                shifted_sample,
            conditional_ps * \
                np.power(shifted_sample, 2)])

        return -s_bar

    def _compute_M_step(self, params, action):

        # Extract each natural parameter estimate for each component
        s_0 = params[:self.K]
        s_1 = params[self.K:2*self.K]
        s_2 = params[2*self.K:]

        self.ps = np.copy(s_0)

        if action == 0:
            # Calculate raw M step
            mus = s_1 / s_2
            sigmas = (s_2 - s_1) / s_0

            # Normalize for baseline effect
            self.mus = mus - self.baseline_mu
            self.sigmas = sigmas - self.baseline_sigma

    def get_projection(self, data, w):

        return w

    def _get_densities(self, data):

        mus = self.mus + self.baseline_mu
        sigmas = self.sigmas + self.baseline_sigma
        numer = - np.power(mus - data, 2)
        denom = 2 * np.power(sigmas, 2)
        kernel = np.exp(numer / denom)
        normalizer = np.power(
            sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer

class RademacherGaussianMixtureModel:

    def __init__(self, 
        baseline_mu=0,
        baseline_sigma=0):

        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.K = 2
        ps = np.random.randint(low=0, size=self.K)
        self.ps = ps / np.sum(ps)
        self.mus = np.zeros(self.K)
        self.sigmas = np.ones(self.K)

    def get_objective(self, data, params):

        print 'Poop'

    def get_residuals(self, data, params):
        
        print 'Poop'

    def get_coordinate_counts(self, data):

        print 'Poop'

    def get_datum(self, data, i):

        print 'Poop'

    def get_parameter_shape(self, data, params):

        print 'Poop'

    def get_gradient(self, data, params):

        (sample, r_scale) = data
        gradient = self._get_gradient(sample, r_scale)

        self._compute_M_step(params)

        return gradient
        
    def _get_gradient(self, sample, r_scale):

        # Get weights for expected sufficient stats
        shifted_sample = sample + np.hstack(
            [r_scale, -r_scale])
        densities = self._get_densities(
            shifted_sample)
        numers = densities * self.ps
        conditional_ps = numers / np.sum(numers)

        # Get natural parameter/dual space stuff
        # (Expected sufficient stats conditioned on observations)
        s_bar = np.vstack([
            conditional_ps,
            conditional_ps * \
                shifted_sample,
            conditional_ps * \
                np.power(shifted_sample, 2)])

        return -s_bar

    def _compute_M_step(self, params):

        # Extract each natural parameter estimate for each component
        s_0 = params[:self.K]
        s_1 = params[self.K:2*self.K]
        s_2 = params[2*self.K:]

        self.ps = np.copy(s_0)

        # Calculate raw M step
        mus = s_1 / s_2
        sigmas = (s_2 - s_1) / s_0

        # Normalize for baseline effect
        self.mus = mus - self.baseline_mu
        self.sigmas = sigmas - self.baseline_sigma

    def get_projection(self, data, w):

        return w

    def _get_densities(self, data):

        mus = self.mus + self.baseline_mu
        sigmas = self.sigmas + self.baseline_sigma
        numer = - np.power(mus - data, 2)
        denom = 2 * np.power(sigmas, 2)
        kernel = np.exp(numer / denom)
        normalizer = np.power(
            sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer
