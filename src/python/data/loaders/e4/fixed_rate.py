from data.loaders.data_loader import AbstractDataLoader
from data.errors import EOSError

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

    def __init__(self, filepath, window, process_line):

        self.filepath = filepath

        with open(self.filepath) as f:
            self.timestamp = float(f.readline().split(',')[0])
            self.hrtz = float(f.readline().split(',')[0])

        self.pl = process_line
        self.window = window
        self.cols = self.hrtz * self.window

        self.num_rounds = 0
        self.data = None

    def get_datum(self):

        if self.data is None:
            data_list = []
           
            with open(self.filepath) as f:
                data_list = [self.pl(line) for line in f]

            remainder = len(data_list) % self.cols
            n = len(data_list) / self.cols
            data_array = np.array(data_list[:-remainder]) 
            self.data = np.reshape(data_array, (n, self.cols))

        return self.data

    def get_status(self):

        return {
            'hrtz': self.htrz,
            'window': self.window,
            'filepath': self.filepath,
            'num_samples': self.num_samples,
            'num_rounds': self.num_rounds
        }

    def cols(self):

        return self.cols

    def rows(self):

        return self.num_rounds
