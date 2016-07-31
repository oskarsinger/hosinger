import numpy as np

class BatchServer:

    def __init__(self, 
        data_loader, 
        reg=0.1, 
        center=False,
        lazy=True, 
        num_coords=None):

        self.dl = data_loader
        self.lazy = lazy
        self.reg = reg
        self.center = center
        self.num_coords = num_coords

        self.data = None if lazy else self.dl.get_data()

    def get_data(self):

        if self.data is None:
            self.data = self.dl.get_data()

            if self.center:
                self.data -= np.mean(self.data, axis=0)

            if self.num_coords is not None:
                self._avg_data()

        return np.copy(self.data)

    def _avg_data(self):

        new_batch = np.zeros((self.data.shape[0], self.num_coords))
        sample_size = self.cols() / self.num_coords

        for i in xrange(self.num_coords):
            begin = i * sample_size
            end = begin + sample_size

            if end + sample_size > self.data.shape[1]+1:
                new_batch[:,i] = np.mean(self.data[:,begin:], axis=1)
            else:
                new_batch[:,i] = np.mean(self.data[:,begin:end], axis=1)

        self.data = new_batch 

    def cols(self):

        cols = self.dl.cols()

        if self.num_coords is not None:
            cols = self.num_coords

        return cols

    def rows(self):

        return 0 if self.data is None else self.data.shape[0]

    def get_status(self):

        return {
            'data_loader': self.dl,
            'reg': self.reg,
            'lazy': self.lazy,
            'online': False,
            'data': self.data}
