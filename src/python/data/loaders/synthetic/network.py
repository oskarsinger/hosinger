import numpy as np

from drrobert.random import normal
from math import sqrt

class ExposureShiftedGaussianWithBaselineEffectLoader:

    def __init__(self,
        n,
        sign,
        mu,
        sigma,
        neighbors,
        baseline_mu=0,
        baseline_sigma=0):

        self.n = n
        self.sign = sign
        self.mu = mu
        self.sigma = sigma
        self.neighbors = neighbors
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.action = None

    def set_action(self, action):

        self.action = action 

    def get_data(self):

        exposure = self.sign * sum(
            [1 if n.action else 0 
             for n in self.neighbors])**(0.5)
        baseline = normal(
            loc=self.baseline_mu,
            scale=self.baseline_sigma)
        treatment = 0

        if self.action:
            treatment = normal(
                loc=self.mu
                scale=self.sigma)

        return baseline + treatment + exposure

    def cols(self):

        print 'Poop'

    def rows(self):

        print 'Poop'

    def name(self):

        return 'RademacherMixtureModelGaussianLoader'
    
class VertexWithExposureLoader:

    def __init__(self,
        v,
        gamma,
        phi=sqrt,
        F=np.random.normal,
        G=np.random.normal,
        theta_F=None,
        theta_G=None):

        self.v = v
        self.gamma = gamma
        self.phi = phi
        self.F = F
        self.G = G

        if theta_F is None:
            theta_F = np.random.randn()

        self.theta_F = theta_F

        if theta_G is None:
            theta_G = np.random.randn()

        self.theta_G = theta_G

        self.neighbors = None
        self.action = None
        self.num_rounds = 0

    def set_neighbors(self, neighbors):

        self.neighbors = neighbors

    def set_action(self, action):

        self.action = action

    def get_action(self):

        return self.action

    def get_data(self):

        self.num_rounds += 1

        n_actions = [n.get_action() 
                     for n in self.neighbors]
        exposure = self.phi(
            float(sum(n_actions)) / len(self.neighbors))
        Y_0 = self.F(loc=self.theta_F)
        tau = self.G(loc=self.theta_G) if self.v in actions else 0

        return Y_0 + tau + self.gamma * exposure
