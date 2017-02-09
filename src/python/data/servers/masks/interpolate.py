import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as IUS

class Interp1DMask:

    def __init__(self, ds):

        self.ds = ds

    def get_data(self):

        full_y = self.ds.get_data()[:,0]
        full_x = np.arange(full_y.shape[0])
        non_nan_indexes = np.logical_not(np.isnan(full_y))
        non_nan_x = np.copy(full_x[non_nan_indexes])
        non_nan_y = full_y[non_nan_indexes]
        f = IUS(non_nan_x, non_nan_y, k=3)
        interped = f(full_x)[:,np.newaxis]

        return interped

    def cols(self):

        return self.ds.cols()

    def rows(self):

        return self.ds.rows()

    def refresh(self):

        self.ds.refresh()

    def get_status(self):
        
        new_status = {
            'ds': self.ds,}

        for (k, v) in self.ds.get_status().items():
            if k not in new_status:
                new_status[k] = v

        return new_status
