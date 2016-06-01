import numpy as np

from optimization.utils import get_minibatch
from drrobert.ml import get_whitened

class Batch2Minibatch:

    def __init__(self, 
        data_loader, batch_size, 
        random=True, lazy=True, whiten=False):

        self.dl = data_loader
        self.bs = batch_size
        self.random = random
        self.lazy = lazy
        self.whiten = whiten

        self.data = None if self.lazy else self._init_data()
        self.num_rounds = 0

    def get_data(self):

        if self.data is None:
            self._init_data()

        current = None

        if self.random:
            current = np.copy(get_minibatch(self.data, self.bs))
        else:
            begin = self.num_rounds * self.bs
            end = begin + self.bs
            (n, p) = self.data.shape
            current = np.copy(self.data[begin:end,:]) \
                if end <= n else \
                None

        self.num_rounds += 1

        if self.whiten:
            current = get_whitened(current)

        return current

    def _init_data(self):

        self.data = self.dl.get_data()

    def rows(self):

        return self.dl.rows()

    def cols(self):

        return self.dl.cols()

    def get_status(self):

        return {
            'data_loader': self.dl,
            'batch_size': self.bs,
            'data': self.data,
            'num_rounds': self.num_rounds,
            'random': self.random,
            'lazy': self.lazy}
