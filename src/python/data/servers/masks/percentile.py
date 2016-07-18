import numpy as np

class PercentileMask:

    def __init__(self, 
        ds, 
        percentiles,
        unfold=False):

        self.ds = ds
        self.percentiles = percentiles
        self.unfold = unfold

        self.num_rounds = 0

    def get_data(self):

        batch = self.ds.get_data()       

        if not isinstance(batch, MissingData):
            if self.unfold:
                (n, p) = batch.shape
                batch = batch.reshape(n*p,1)

            self.num_rounds += 1

            batch = np.percentile(
                batch, 
                self.percentiles,
                axis=1).T

        return batch

    def refresh(self):

        self.ds.refresh()

    def get_status(self):

        percentile_items = {
            'data_server': self.ds,
            'percentiles': self.percentiles,
            'unfold': self.unfold}.items()
        ds_items = self.ds.get_status().items()

        return dict(percentile_items + ds_items)

    def cols(self):
        
        return len(self.percentiles)

    def rows(self):

        return self.ds.rows()
