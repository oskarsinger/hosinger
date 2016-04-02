from data_loader import AbstractDataLoader

import os

import numpy as np

class SmartWatchWindowLoader(AbstractDataLoader):

    def __init__(self, data_dir, window):

        self.data_dir = data_dir
        self.window = window

        self.num_rounds = 0

    def get_datum(self):

        self.num_rounds += 1

        for i in range(

    def get_status(self):

        return {
            'window': self.window,
            'data_dir': self.data_dir,
            'num_rounds': self.num_rounds
        }

    def _prep_file_itr(self, filepath):

        print "Some stuff"
