import numpy as np

from drrobert.random import normal

class RademacherMixtureModelGaussianLoader:

    def __init__(self,
        n,
        ps,
        mus,
        sigmas,
        baseline_mu=0,
        baseline_sigma=0):

        self.n = n
        self.ps = p
        self.mus = mus
        self.sigmas = sigmas
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma

    def get_data(self):

        print 'Poop'
    
class MixtureModelGaussianLoader:

    def __init__(self, 
        n,
        ps,
        mus,
        sigmas):

        self.ps = ps
        self.mus = mus
        self.sigmas = sigmas
        self.n = n

        self.num_components = len(self.ps)
        self.components = np.random.choice(
            self.num_components,
            self.n,
            p=self.ps)[:,np.newaxis]
        self.data = self._get_data_from_components(
            self.components)

    def get_data(self):

        return np.copy(self.data)

    def _get_data_from_components(self, components):

        data = np.zeros_like(components) 

        for i in xrange(self.num_components):
            num_samples = data[data == i].shape[0]
            mu = self.mus[i]
            sigma = self.sigmas[i]

            data[data == i] = normal(
                loc=mu, 
                scale=sigma, 
                shape=(num_samples,1))

        return data

    def name(self):

        return 'MixtureModelGaussianLoader'

    def cols(self):

        return 1

    def rows(self):

        rows = None

        return self.n

