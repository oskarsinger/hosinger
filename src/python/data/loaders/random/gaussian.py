from data_loader import AbstractDataLoader

from numpy.random import randn

class GaussianLoader(AstractDataLoader):

    def __init__(self, n, p):

        self.n = n
        self.p = p

    def get_datum(self):

        return randn(n, p)

    def get_status(self):

        return {
            'n': n,
            'p': p}
