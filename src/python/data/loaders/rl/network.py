import numpy as np

from drrobert.random import normal

class ExposureShiftedGaussianWithBaselineEffectLoader:

    def __init__(self,
        sign,
        mu,
        sigma,
        id_number,
        baseline_mu=0,
        baseline_sigma=0):

        self.sign = sign
        self.mu = mu
        self.sigma = sigma
        self.id_number = id_number
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.action = None
        self.a_history = []
        self.num_rounds = 0

    def set_neighbors(self, neighbors):

        self.neighbors = neighbors
        self.neighbor_actions = {n.id_number : 0
                                 for n in self.neighbors}

    def set_action(self, action):

        self.a_history.append(action)

        for n in self.neighbors:
            n.set_neighbor_action(
                int(action), self.id_number)

    def set_neighbor_action(self, action, neighbor_id):

        self.neighbor_actions[neighbor_id] = action

    def get_reward(self):

        self.num_rounds += 1
        exposure = 0

        if len(self.neighbors) > 0:
            n_actions = self.neighbor_actions.values()
            neighbor_treatment_count = sum(
                [float(na) for na in n_actions])**(0.5)
            ratio = neighbor_treatment_count / len(self.neighbors)
            exposure = ratio**(0.5)

        baseline = normal(
            loc=self.baseline_mu,
            scale=self.baseline_sigma)
        treatment = 0

        if self.a_history[-1]:
            treatment = normal(
                loc=self.mu,
                scale=self.sigma)

        reward = baseline + \
            treatment + \
            self.sign * exposure

        return (reward, exposure)

    def cols(self):

        return 1

    def rows(self):

        return self.num_rounds

    def name(self):

        return 'RademacherMixtureModelGaussianLoader'

# TODO: This is quite abstract. Maybe don't need it.
class VertexWithExposureLoader:

    def __init__(self,
        v,
        gamma,
        phi=lambda x: x**(0.5),
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

    # TODO: figure out why I needed this; for dynamic network structure?
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
