import numpy as np

from drrobert.data_structures import FixedLengthQueue as FLQ
from drrobert.ml import get_pca

class Minibatch2Minibatch:

    def __init__(self, 
        data_loader, batch_size, 
        n_components=None):

        self.dl = data_loader
        self.bs = batch_size
        self.n_components = n_components

        self.num_rounds = 0
        self.data = None
        self.minibatch = FLQ(self.bs)

    def get_data(self):

        self.num_rounds += 1

        return self._get_minibatch()

    def _get_minibatch(self):

        if self.data is None:
            self.data = self.dl.get_data()

        n = self.data.shape[0]
        need = max([self.bs - self.minibatch.get_length(), 1])

        for i in xrange(min([n,need])):
            self.minibatch.enqueue(self.data[i,:])

        if n <= need:
            self.data = None
        else:
            self.data = self.data[need:,:]

        if not self.minibatch.is_full():
            return self._get_minibatch()
        else:
            items = self.minibatch.get_items()
            batch = np.array(items)

            if self.n_components is not None:
                batch = get_pca(
                    batch, n_components=self.n_components)

            return batch

    def rows(self):
        
        return self.num_rounds

    def cols(self):

        return self.dl.cols()

    def get_status(self):

        return {
            'data_loader': self.dl,
            'batch_size': self.bs,
            'n_components': self.n_components,
            'minibatch': self.minibatch,
            'data': self.data,
            'num_rounds': self.num_rounds}
