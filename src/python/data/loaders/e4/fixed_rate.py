from data.loaders.data_loader import AbstractDataLoader
from data.errors import EOSError

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

    def __init__(self, filepath, window, process_line):

        self.filepath = filepath
        self.data = []

        with open(self.filepath) as f:
            self.timestamp = float(self.f.readline().split(',')[0])
            self.hrtz = float(self.f.readline().split(',')[0])
            self.data = [process_line(line)
                         for line in f]

        self.window = window
        self.num_samples = self.hrtz * self.window
        self.num_rounds = 0

    def get_datum(self):
       
        self.num_rounds += 1


    def get_status(self):

        return {
            'hrtz': self.htrz,
            'window': self.window,
            'filepath': self.filepath,
            'num_samples': self.num_samples,
            'num_rounds': self.num_rounds
        }

    def cols(self):

        return self.window

    def rows(self):

        return self.num_rounds
