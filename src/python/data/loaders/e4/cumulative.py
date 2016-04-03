from data_loader import AbstractDataLoader
from global_utils.file_io import list_dirs_only

import numpy as np

class SmartWatchWindowLoader(AbstractDataLoader):

    def __init__(self, data_dir, window):

        self.top_dir = data_dir
        self.sub_dirs = list_dirs_only(self.top_dir).sort()
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

    def _prep_itr_for_dir(self, filepath):

        print "Some stuff"
