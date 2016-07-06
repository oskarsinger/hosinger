import numpy as np

class Data2Percentiles:

    def __init__(self, 
        ds, 
        percentiles=np.linspace(0,1,num=10),
        unfold=False):

        self.ds = ds
        self.percentiles = percentiles
        self.unfold = unfold

        self.num_rounds = 0

    def get_data(self):

        batch = ds.get_data()       

        if self.unfold:
            (n, p) = batch.shape
            batch = batch.reshape(n*p,1)

        self.num_rounds += 1

        return np.percentile(
            batch, 
            self.percentiles,
            axis=1).T

    def get_status(self):

        return {
            'data_server': self.ds,
            'percentiles': self.quantiles,
            'unfold': self.unfold,
            'num_rounds': self.num_rounds}
