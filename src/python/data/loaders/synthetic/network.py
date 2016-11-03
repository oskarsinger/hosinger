import numpy as np

class VertexWithExposureLoader:

    def __init__(self,
        v,
        gamma,
        phi,
        F,
        G,
        theta):

        self.v = v
        self.gamma = gamma
        self.phi = phi
        self.F = F
        self.G = G
        self.theta = theta

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
        Y_0 = self.F(self.theta)
        tau = self.G(self.theta) if self.v in actions else 0

        return Y_0 + tau + self.gamma * exposure
