import numpy as np

from drrobert.misc import unzip
from scipy.stats import norm
from optimization.utils import get_simplex_projection

# TODO: consider feeding in get_action as an arg
class BanditNetworkRademacherGaussianMixtureModel:

    def __init__(self, 
        budget,
        id_number,
        baseline_mu=0,
        baseline_sigma=0):

        self.budget = budget
        self.id_number = id_number
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.K = 2
        self.p = self.K * 3
        self.ps = np.random.dirichlet(
            np.random.randn(self.K))
        self.mus = np.zeros(self.K)
        self.sigmas = np.ones(self.K)

    def get_action(self, params):

        max_id = int(params.shape[0] / self.p)
        Sk = np.random.choice(
            max_id,
            replace=False,
            size=self.budget).tolist()

        return int(self.id_number in Sk)

    def get_objective(self, data, params):

        objective = 0

        for ((reward, r_scale), action) in data:
            shifted_sample = reward + np.hstack(
                [r_scale, -r_scale])
            densities = self._get_densities(
                shifted_sample)
            likelihood = np.dot(densities, self.ps)

            objective += likelihood

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

        (samples_and_r_scales, actions) = unzip(data)
        (samples, r_scales) = unzip(samples_and_r_scales)
        gradient = self._get_gradient(
            samples, r_scales, actions)

        self._compute_M_step(params, actions)

        return gradient
        
    def _get_gradient(self, samples, r_scales, actions):

        # Get weights for expected sufficient stats
        r_scales = np.array(r_scales)[:,np.newaxis]
        print 'r_scales', np.any(np.isnan(r_scales))
        actions = np.array(actions)[:,np.newaxis]
        print 'actions', np.any(np.isnan(actions))
        shifted_samples = np.array(samples) + \
            np.hstack([r_scales, -r_scales])
        print 'shifted_samples', np.any(np.isnan(shifted_samples))
        densities = self._get_densities(
            shifted_samples)
        print 'densities', np.any(np.isnan(densities))
        numers = densities * self.ps.T
        print 'numers', np.any(np.isnan(numers))
        conditional_ps = numers / np.sum(numers)
        print 'conditional_ps', conditional_ps

        # TODO: make sure this is the correct way to update when no treatment is applied
        index = np.hstack([actions==0,actions==0])
        shifted_samples[index] = 0

        # Get natural parameter/dual space stuff
        # (Expected sufficient stats conditioned on observations)
        s1_hat = np.mean(
            conditional_ps, 
            axis=0)[:,np.newaxis]
        print 's1_hat', np.any(np.isnan(s1_hat))
        s2_hat = np.mean(
            conditional_ps * shifted_samples,
            axis=0)[:,np.newaxis]
        print 's2_hat', np.any(np.isnan(s2_hat))
        s3_hat = np.mean(
            conditional_ps * np.power(shifted_samples, 2),
            axis=0)[:,np.newaxis]
        print 's3_hat', np.any(np.isnan(s3_hat))
        s_bar = np.vstack([s1_hat, s2_hat, s3_hat])

        return -s_bar

    def _compute_M_step(self, params, action):

        # Extract each natural parameter estimate for each component
        s_0 = get_simplex_projection(params[:self.K])

        self.ps = np.copy(s_0)

        if not np.any(self.ps == 0):
            s_1 = params[self.K:2*self.K]
            s_2 = params[2*self.K:]

            s_1[s_1 >= s_2] = s_2[s_1 >= s_2] - 1


            # TODO: should I do this even when action is 0? Does it matter?
            # Calculate raw M step
            mus = s_1 / s_2
            print 'mus', mus
            sigmas = (s_2 - s_1) / s_0
            print 'sigmas', sigmas

            # Normalize for baseline effect
            self.mus = mus - self.baseline_mu
            self.sigmas = sigmas - self.baseline_sigma

    def get_projection(self, data, w):

        return w

    def _get_densities(self, samples):

        mus = self.mus + self.baseline_mu
        sigmas = self.sigmas + self.baseline_sigma
        numer = - np.power(mus.T - samples, 2)
        denom = 2 * np.power(sigmas, 2)
        kernel = np.exp(numer / denom.T)
        normalizer = np.power(
            sigmas * 2 * np.pi, 0.5)

        return kernel / normalizer.T

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
