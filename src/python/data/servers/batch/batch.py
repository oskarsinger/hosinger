import numpy as np

class BatchServer:

    def __init__(self, 
        data_loader, 
        reg=0.1, 
        center=False,
        num_coords=None):

        self.dl = data_loader
        self.reg = reg
        self.center = center
        self.num_coords = num_coords

    def get_data(self):

        data = self.dl.get_data()

        if self.center:
            data -= np.mean(data, axis=0)

        if self.num_coords is not None:
            self._avg_data(data)

        return data

    def _avg_data(self, data):

        new_batch = np.zeros((data.shape[0], self.num_coords))
        sample_size = self.cols() / self.num_coords

        for i in xrange(self.num_coords):
            begin = i * sample_size
            end = begin + sample_size

            if end + sample_size > data.shape[1]+1:
                new_batch[:,i] = np.mean(data[:,begin:], axis=1)
            else:
                new_batch[:,i] = np.mean(data[:,begin:end], axis=1)

        return new_batch 

    def cols(self):

        cols = self.dl.cols()

        if self.num_coords is not None:
            cols = self.num_coords

        return cols

    def rows(self):

        return self.dl.rows()

    def name(self):

        return self.dl.name()

    def refresh(self):

        'Poop'

        # TODO: fill this in later

    def get_status(self):

        return {
            'data_loader': self.dl,
            'reg': self.reg,
            'online': False}
