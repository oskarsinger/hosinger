import numpy as np

from drrobert.data_structures import FixedLengthQueue as FLQ
from drrobert.ml import get_pca
from data.missing import MissingData

class Minibatch2Minibatch:

    def __init__(self, 
        data_loader, batch_size, 
        num_coords=None):

        self.dl = data_loader
        self.bs = batch_size
        self.num_coords = num_coords

        self.num_rounds = 0
        self.data = None
        self.minibatch = FLQ(self.bs)
        self.num_missing_rows = 0

    def get_data(self):

        self.num_rounds += 1

        return self._get_minibatch()

    def _get_minibatch(self):

        if self.data is None:
            self.data = self.dl.get_data()

            if isinstance(self.data, MissingData):
                self.num_missing_rows = self.data.get_status()['num_missing_rows']

        batch = None

        # TODO: would be great if this could be cleaned up a bit
        if not isinstance(self.data, MissingData):
            n = self.data.shape[0]
            need = max([self.bs - self.minibatch.get_length(), 1])

            for i in xrange(min([n,need])):
                self.minibatch.enqueue(np.copy(self.data[i,:]))

            self.data = None if n <= need else self.data[need:,:]

            if not self.minibatch.is_full():
                batch = self._get_minibatch()
            else:
                items = self.minibatch.get_items()
                data_array = np.array(items)
                batch = data_array \
                    if self.num_coords is None else \
                    self._get_avgd(data_array)
        elif self.num_missing_rows > 0:
            batch = self.data
            self.num_missing_rows -= 1 
        else:
            self.data = None
            batch = self._get_minibatch()

        return batch

    # TODO: consider moving this to drrobert
    def _get_avgd(self, batch):

        new_batch = np.zeros((self.bs, self.num_coords))
        sample_size = self.cols() / self.num_coords

        for i in xrange(self.num_coords):
            begin = i * sample_size
            end = begin + sample_size

            if end + sample_size > batch.shape[1]+1:
                new_batch[:,i] = np.mean(batch[:,begin:], axis=1)
            else:
                new_batch[:,i] = np.mean(batch[:,begin:end], axis=1)

        return new_batch

    def rows(self):
        
        return self.num_rounds

    def cols(self):

        cols = self.dl.cols()

        if self.num_coords is not None:
            cols = self.num_coords

        return cols

    def refresh(self):

        self.dl.refresh()
        self.minibatch = FLQ(self.bs)
        self.data = None
        self.num_rounds = 0

    def get_status(self):

        return {
            'data_loader': self.dl,
            'batch_size': self.bs,
            'minibatch': self.minibatch,
            'data': self.data,
            'online': True,
            'num_rounds': self.num_rounds}
