import numpy as np

class VertexWithExposureLoader:

    def __init__(self,
        v,
        gamma,
        phi,
        F,
        G,
        theta,
        neighbors):

        self.v = v
        self.gamma = gamma
        self.phi = phi
        self.F = F
        self.G = G
        self.theta = theta
        self.neigbors = neighbors

        self.num_rounds = 0

    def get_data(self, actions):

        self.num_rounds += 1

        d_e = float(sum(
            [a in self.neighbors for a in actions]))
        exposure = self.phi(d_e / len(self.neighbors))
        Y_0 = self.F(self.theta)
        tau = self.G(self.theta) if self.v in actions else 0

        return Y_0 + tau + self.gamma * exposure
