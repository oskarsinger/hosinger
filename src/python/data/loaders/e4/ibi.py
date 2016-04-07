from data_loader import AbstractDataLoader
from errors import EOSError

import os

import numpy as np

class IBILoader(AbstractDataLoader):

    def __init__(self, filepath, window, process_line):

        self.filepath = filepath
        self.f = open(filepath)
        self.window = window

        self.timestamp = float(self.f.readline().split(',')[0])
        self.num_rounds = 0

    def get_datum(self):

        self.num_rounds += 1

        batch = []
