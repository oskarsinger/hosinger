import numpy as np

from data.utils import get_minibatch
from drrobert.ml import get_pca

class Batch2Minibatch:

    def __init__(self, 
        batch_size, 
        data_loader=None,
        data_server=None,
        center=False,
        random=True, 
        lazy=True,
        n_components=None):

        if data_loader is None:
            data_loader = data_server.get_status()['data_loader']

        if data_server is None:
            data_server = data_loader

        self.ds = data_server
        self.dl = data_loader
        self.bs = batch_size
        self.center = center
        self.random = random
        self.lazy = lazy
        self.n_components = n_components

        self.data = None if self.lazy else self._init_data()
        self.num_rounds = 0

    def get_data(self):

        if self.data is None:
            self._init_data()

            if self.center:
                self.data -= np.mean(self.data, axis=0)

        current = None

        if self.random:
            current = np.copy(get_minibatch(self.data, self.bs))
        else:
            n = self.data.shape[0]
            begin = (self.num_rounds * self.bs) % n
            end = begin + self.bs
            
            if end > n:
                begin = 0
                end = self.bs

            current = np.copy(self.data[begin:end,:])

        self.num_rounds += 1

        if self.n_components is not None:
            current = get_pca(current, n_components=self.n_components)

        return current

    def _init_data(self):

        self.data = self.ds.get_data()
        self.num_batches = int(float(self.data.shape[0]) / self.bs)

    def finished(self):

        finished = False

        if self.data is not None:
            next_num_samples = (self.num_rounds + 1) * self.bs
            n = self.data.shape[0]
            finished = next_num_samples > n

        return finished

    def rows(self):

        return self.ds.rows()

    def cols(self):

        return self.ds.cols()

    def refresh(self):

        self.ds.refresh()
        self.data = None
        self.num_rounds = 0

    def get_status(self):

        return {
            'data_server': self.ds,
            'data_loader': self.dl,
            'batch_size': self.bs,
            'n_components': self.n_components,
            'data': self.data,
            'num_rounds': self.num_rounds,
            'random': self.random,
            'lazy': self.lazy}
