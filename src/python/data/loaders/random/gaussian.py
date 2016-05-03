from data.loaders import AbstractDataLoader
from linal.random.utils import get_rank_k

from numpy.random import randn

class GaussianLoader(AbstractDataLoader):

    def __init__(self, n, p, k=None):

        self.n = n
        self.p = p
        self.t = 0

        if k is None:
            self.low_rank = False
        else:
            if k > min([n, p]):
                raise ValueError(
                    'Parameter k must not exceed the minimum matrix dimension.')
            else:
                self.low_rank = True

        self.k = k

    def get_datum(self):

        self.t += 1

        X = None

        if self.low_rank:
            X = get_rank_k(self.n, self.p, self.k)
        else:
            X = randn(self.n, self.p)

        return X

    def get_status(self):

        return {
            'n': self.n,
            'p': self.p,
            't': self.t,
            'k': self.k}

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n * self.t
