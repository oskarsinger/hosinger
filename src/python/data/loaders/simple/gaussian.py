import numpy as np

from linal.svd_funcs import get_svd_power
from linal.random.utils import get_rank_k
from drrobert.misc import get_checklist

class GaussianLoader:

    def __init__(self, 
        n, p, 
        lazy=True,
        batch_size=None, 
        k=None, 
        mean=None):

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

        self.batch_size = batch_size

        self.n = n
        self.p = p
        self.k = k
        self.lazy = lazy

        # Set mean of each column by input constants
        if mean is not None:
            if not mean.shape[1] == self.p:
                raise ValueError(
                    'Length of means parameter must be equal to p.')

        self.mean = mean

        # Generate data
        self.X = None
        
        if not self.lazy:
            self.X = _get_batch(self.n, self.p, self.k, self.mean)
            
        # Checklist for which rows sampled in current epoch
        self.sampled = get_checklist(xrange(self.n))

        # Number of requests made for data
        self.num_rounds = 0

        # Number of times through the full data set
        self.num_epochs = 0

    def get_data(self):

        if self.X is None:
            self.X = _get_batch(self.n, self.p, self.k, self.mean)

        self.num_rounds += 1

        data = None

        if self.batch_size is None:
            data = np.copy(self.X)
        else:
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

            data = np.copy(self.X[sample_indexes,:])

        return data

    def get_status(self):

        return {
            'n': self.n,
            'p': self.p,
            'num_rounds': self.num_rounds,
            'k': self.k,
            'batch_size': self.batch_size,
            'sampled': self.sampled,
            'lazy': self.lazy,
            'low_rank': self.low_rank,
            'mean': self.mean}

    def name(self):

        return 'GaussianData'

    def finished(self):

        finished = False

        if self.batch_size is None:
            finished = self.num_rounds > 1
        else:
            num_unsampled = sum(
                [0 if s else 1 for s in self.sampled.values()])
            finished = num_unsampled < self.batch_size
            
        return finished

    def refresh(self):

        self.X = None
        self.sampled = get_checklist(xrange(self.n))
        self.num_rounds = 0

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n

class ShiftingMeanGaussianLoader:

    def __init__(self, 
        p, mean, rate, 
        max_rounds=1000,
        batch_size=1, 
        k=None, 
        variance=None):

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
        self.max_rounds = max_rounds
        self.bs = batch_size

        # Figure out clean way to verify dimensional validity
        self.rate = rate

        if variance is None:
            variance = np.identity(self.p)

        self.variance = variance

        # Set mean of each column by input constants
        if not mean.shape[1] == self.p:
            raise ValueError(
                'Length of means parameter must be equal to p.')

        self.mean = mean

        # Number of requests made for data
        self.num_rounds = 0

    def get_data(self):

        # Calculate current means
        scale = get_svd_power(self.rate, self.num_rounds)
        current_mean = np.dot(self.mean, scale)

        # Get batch
        batch = _get_batch(self.bs, self.p, self.k, current_mean)

        # Update global state variable
        self.num_rounds += 1

        return batch

    def finished(self):

        return self.num_rounds >= self.max_rounds

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'p': self.p,
            'num_rounds': self.num_rounds,
            'mean': self.mean,
            'rate': self.rate,
            'k': self.k,
            'batch_size': self.bs}

    def name(self):

        return 'ShiftingMeanGaussianData'

    def cols(self):

        return self.p

    def rows(self):

        return self.num_rounds

def _get_batch(bs, p, k=None, mean=None):

    batch = None

    if k is not None:
        batch = get_rank_k(bs, p, k)
    else:
        batch = np.random.randn(bs, p)

    if mean is not None:
        batch += mean

    return batch
