import numpy as np

class BatchServer:

    def __init__(self, data_loader, reg=0.1, lazy=True):

        self.dl = data_loader
        self.lazy = lazy
        self.reg = reg

        self.data = None if lazy else self.dl.get_data()

    def get_data(self):

        if self.data is None:
            self.data = self.dl.get_data()

        return np.copy(self.data)

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
