import numpy as np

class BernoulliSubsetServer:

    def __init__(self, bernoulli_ps):

        self.bernoulli_ps = bernoulli_ps

        self.num_nodes = self.bernoulli_ps.shape[0]
        self.choices = [0, 1]
        self.ps = np.hstack(
            [1-bernoulli_ps, bernoulli_ps])
        self.get_sample = lambda n: np.random.choice(
            self.choices, p=self.ps[n,:])

    def get_subset(self):

        return [n for n in range(self.num_nodes)
                if self.get_sample(n) == 1]
