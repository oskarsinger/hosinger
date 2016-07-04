import numpy as np

class BatchServer:

    def __init__(self, 
        data_loader, 
        reg=0.1, 
        lazy=True, 
        num_coords=None):

        self.dl = data_loader
        self.lazy = lazy
        self.reg = reg
        self.num_coords = num_coords

        self.data = None if lazy else self.dl.get_data()

    def get_data(self):

        if self.data is None:
            self.data = self.dl.get_data()

            if self.num_coords is not None:
                self._avg_data()

        return np.copy(self.data)

    def _get_avgd(self):

        new_batch = np.zeros((self.data.shape[0], self.num_coords))
        sample_size = self.cols() / self.num_coords

        for i in xrange(self.num_coords):
            begin = i * sample_size
            end = begin + sample_size

            if end + sample_size > batch.shape[1]+1:
                new_batch[:,i] = np.mean(self.data[:,begin:], axis=1)
            else:
                new_batch[:,i] = np.mean(self.data[:,begin:end], axis=1)

        return new_batch

    def cols(self):

        return self.dl.cols()

    def rows(self):

        return 0 if self.data is None else self.data.shape[0]

    def get_status(self):

        return {
            'data_loader': self.dl,
            'reg': self.reg,
            'lazy': self.lazy,
            'data': self.data}
