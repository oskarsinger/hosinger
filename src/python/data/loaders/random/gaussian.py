from data.loaders import AbstractDataLoader
from linal.random.utils import get_rank_k
from drrobert.misc import get_checklist
from drrobert.random import normal

import numpy as np

class GaussianLoader(AbstractDataLoader):

    def __init__(self, n, p, batch_size=None, k=None, means=None):

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

        # Set mean of each column by input constants
        if means is not None:
            if not len(means) == self.p:
                raise ValueError(
                    'Length of means parameter must be equal to p.')

        self.means = means

        # Generate data
        self.X = _get_batch(self.n, self.p, self.k, self.means)
            
        # Checklist for which rows sampled in current epoch
        self.sampled = get_checklist(xrange(self.n))

        # Number of requests made for data
        self.num_rounds = 0

        # Number of times through the full data set
        self.num_epochs = 0

    def get_data(self):

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
            'low_rank': self.low_rank,
            'means': self.means}

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n

class ShiftingMeanGaussianLoader(AbstractDataLoader):

    def __init__(self, p, means, rate, batch_size=1, k=None):

        # Check validity of k parameter
        if k is None:
            self.low_rank = False
        else:
            if k > min([n, p]):
                raise ValueError(
                    'Parameter k must not exceed the minimum matrix dimension.')
            else:
                self.low_rank = True

        self.p = p
        self.k = k
        self.bs = batch_size
        self.rate = rate

        # Set mean of each column by input constants
        if not len(means) == self.p:
            raise ValueError(
                'Length of means parameter must be equal to p.')

        self.means = means

        # Number of requests made for data
        self.num_rounds = 0

    def get_data(self):

        # Calculate current means
        scale = (self.rate)**(self.num_rounds)
        current_means = [scale * mu for mu in means]

        # Get batch
        batch = _get_batch(self.bs, self.p, self.k, current_means)

        # Update global state variable
        self.num_rounds += 1

        return batch

    def get_status(self):

        return {
            'p': self.p,
            'num_rounds': self.num_rounds,
            'means': self.means,
            'rate': self.rate,
            'k': self.k,
            'batch_size': self.bs}

    def cols(self):

        return self.p

    def rows(self):

        return self.num_rounds

def _get_batch(bs, p, k, means):

    batch = None

    if k is not None:
        batch = get_rank_k(bs, p, k)
    else:
        batch = np.random.randn(bs, p)

    if means is not None:
        for i in range(p):
            batch[:,i] += means[i]

    return batch
