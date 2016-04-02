from data_loader import AbstractDataLoader
from errors import EOSError

import os

import numpy as np

class SensorLoader(AbstractDataLoader):

    def __init__(self, filepath, window, process_line):

        self.filepath = filepath
        self.f = open(filepath)
        self.window = window

        self.timestamp = float(self.f.readline().split(',')[0])
        self.hrtz = float(self.f.readline().split(',')[0])
        self.num_samples = self.hrtz * self.window
        self.num_rounds = 0

    def get_datum(self):
       
        self.num_rounds += 1

        batch = []

        for i in range(self.num_samples):

            val = None

            try:
                val = process_line(self.f.readline())
            except EOFError:
                raise EOSError(
                    'Reached end of sensor file: ' + filepath + '.')

            batch.extend(val)

        return np.array(batch)

    def get_status(self):

        return {
            'hrtz': self.htrz,
            'window': self.window,
            'filepath': self.filepath,
            'num_samples': self.num_samples,
            'num_rounds': self.num_rounds
        }
