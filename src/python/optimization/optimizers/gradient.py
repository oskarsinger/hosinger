import utils as ou

class GradientOptimizer:

    def __init__(self, beta=None, dual_avg=False):

        if beta is None:
            beta = 0
            self.alpha = 1
        else:
            self.alpha = 1 - beta

        self.beta = beta
        self.dual_avg = dual_avg
        self.search_direction = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.search_direction = ou.get_search_direction(
            self.search_direction, 
            gradient, 
            self.dual_avg,
            alpha=self.alpha, 
            beta=self.beta)

        return parameters - eta * self.search_direction

    def get_status(self):

        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'dual_avg': self.dual_avg,
            'search_direction': self.search_direction,
            'num_rounds': self.num_rounds}
