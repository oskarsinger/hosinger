from data.loaders import AbstractDataLoader

import os

import numpy as np

class IEILoader(AbstractDataLoader):

    def __init__(self, dir_path, filename, seconds, process_line):

        self.dir_path = dir_path
        self.filename = filename
        self.timestamps = []

        subdirs = fio.list_dirs_only(dir_path)

        for subdir in subdirs:
            filepath = os.path.join(subdir, self.filename)

            with open(filepath) as f:
                self.timestamps.append((
                    filepath, 
                    float(f.readline().split(',')[0])))

        self.timestamps = sorted(
            self.timestamps, key=lambda x: x[1])
        self.pl = process_line
        self.seconds = seconds
        self.window = self.hertz * self.seconds
        self.num_rounds = 0
        self.data = None

    def get_datum(self):

        self.num_rounds += 1

        batch = []

    def cols(self):

        return self.window

    def rows(self):

        return self.num_rounds
