from data.loaders.data_loader import AbstractDataLoader
from data.errors import EOSError
from global_utils import file_io as fio

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

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
                self.hertz = float(f.readline().split(',')[0])

        self.timestamps = sorted(
            self.timestamps, key=lambda x: x[1])
        self.pl = process_line
        self.seconds = seconds
        self.window = self.hertz * self.seconds
        self.num_rounds = 0
        self.data = None

    def get_datum(self):

        if self.data is None:
            data_list = []
           
            for (fp, ts) in self.timestamps:
                with open(fp) as f:
                    f.readline() # timestamp
                    f.readline() # frequency
                    data_list.extend([self.pl(line) for line in f])

            remainder = int(len(data_list) % self.window)
            n = len(data_list) / self.window

            if remainder > 0:
                data_list = data_list[:-remainder]

            data_array = np.array(data_list) 
            self.data = np.reshape(data_array, (n, self.window))

        return self.data

    def get_status(self):

        return {
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'timestamps': self.filepaths
        }

    def cols(self):

        return self.window

    def rows(self):

        return self.data.shape[0] if data is not None else 0
