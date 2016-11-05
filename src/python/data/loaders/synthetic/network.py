import numpy as np

from math import sqrt

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
