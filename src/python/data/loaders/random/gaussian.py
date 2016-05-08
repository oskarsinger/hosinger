from data.loaders import AbstractDataLoader
from linal.random.utils import get_rank_k
from global_utils.misc import get_checklist

import numpy as np

class GaussianLoader(AbstractDataLoader):

    def __init__(self, n, p, batch_size=None, k=None):

        # Check validity of k parameter
        if k is None:
            self.low_rank = False
        else:
            if k > min([n, p]):
                raise ValueError(
                    'Parameter k must not exceed the minimum matrix dimension.')
            else:
                self.low_rank = True

        # Check validity of batch_size parameter
        if batch_size is not None:
            if batch_size > n:
                raise ValueError(
                    'Parameter batch_size must not exceed parameter n.')
            else:
                self.batch_size = batch_size
        else:
            self.batch_size = n

        self.n = n
        self.p = p
        self.k = k

        if self.low_rank:
            self.X = get_rank_k(self.n, self.p, self.k)
        else:
            self.X = np.random.randn(self.n, self.p)
            
        # Checklist for which rows sampled in current epoch
        self.sampled = get_checklist(xrange(self.n))

        # Number of requests made for data
        self.num_rounds = 0

        # Number of times through the full data set
        self.num_epochs = 0

    def get_datum(self):

        self.num_rounds += 1

        # Check for the rows that have not been sampled this epoch
        unsampled = [i for (i, s) in self.sampled.items() if not s]

        # Refresh if unsampled will not fill a batch
        if len(unsampled) < self.batch_size:
            self.sampled = get_checklist(xrange(self.n))
            self.num_epochs += 1

            unsampled = self.sampled.keys()

        # Sample indexes corresponding to rows in data matrix
        sample_indexes = np.random.choice(
            np.array(unsampled), self.batch_size, replace=False)
        
        # Update checklist with sampled rows
        for i in sample_indexes.tolist():
            self.sampled[i] = True

        return np.copy(self.X[sample_indexes,:])

    def get_status(self):

        return {
            'n': self.n,
            'p': self.p,
            'num_rounds': self.num_rounds,
            'k': self.k,
            'batch_size': self.batch_size,
            'sampled': self.sampled,
            'low_rank': self.low_rank}

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n
