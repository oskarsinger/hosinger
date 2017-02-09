import h5py
import os

import numpy as np

from drrobert.file_io import get_timestamped as get_ts

class CCAPairwiseServer:

    def __init__(self,
        ds1, ds2,
        save_load_path,
        k=1,
        load=False,
        save=False):

        self.ds1 = ds1
        self.ds2 = ds2
        self.save_load_path = save_load_path
        self.k = k
        self.load = load
        self.save = save

    def get_data(self):

        print 'Poop'

    def refresh(self):

        self.ds1.refresh()
        self.ds2.refresh()

        self.num_rounds = 0

    def get_status(self):

        return {
            'ds1': ds1,
            'ds2': ds2,
            'save_load_path': save_load_path,
            'k': k,
            'load': load,
            'save': save}
