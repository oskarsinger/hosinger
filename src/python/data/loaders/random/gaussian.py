from data.loaders import AbstractDataLoader

from numpy.random import randn

class GaussianLoader(AbstractDataLoader):

    def __init__(self, n, p):

        self.n = n
        self.p = p
        self.t = 0

    def get_datum(self):

        self.t += 1

        return randn(self.n, self.p)

    def get_status(self):

        return {
            'n': n,
            'p': p}

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n * self.t
