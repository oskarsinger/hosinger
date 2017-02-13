import numpy as npimport numpy as np

from linal.utils.product import get_quadratic

class QuadraticZeroOrderLoader:

    def __init__(self,
        p=2,
        A=None,
        noise_mean=None):

        self.p = p

        if A is None:
            A = np.eye(p)

        self.A = A
        self.noise_mean = noise_mean
        
        self.num_rounds = 0

    def get_data(self, X):

        self.num_rounds += 1

        data = get_quadratic(X, self.A)

        if self.noise_mean is not None:
            noise = np.random.randn() + self.noise_mean
            data += noise

        return data

    def name(self):

       return 'QuadraticZeroOrderLoader'

    def finished(self):

       return False

    def refresh(self):

        self.num_rounds = 0

    def cols(self):
        
        return 1

    def rows(self):

        return self.num_rounds

    def get_status(self):

        return {
            'p': p,
            'A': A,
            'noise_mean': noise_mean,
            'num_rounds': num_rounds}
