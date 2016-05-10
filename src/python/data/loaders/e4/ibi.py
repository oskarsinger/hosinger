from data.loaders import AbstractDataLoader
from global_utils import file_io as fio

import os

import numpy as np

class IBILoader(AbstractDataLoader):

    def __init__(self, dir_path, filename, seconds, process_line):

        self.dir_path = dir_path
        self.filename = filename
        self.pl = process_line
        self.seconds = seconds

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
        self.num_rounds = 0
        self.rounds_per_file = [0]
        self.data = self._get_file_data()

    def get_datum(self):

        self.num_rounds += 1
        self.rounds_per_file[-1] += 1
        
        # The current time window of interest
        # This is relative to current file's start time
        threshold = self.rounds_per_file[-1] * self.seconds

        # Inter-beat intervals to return
        ibis = []
    
        for (time, ibi) in self.data:

            if time > threshold:
                break
            
            ibis.append(ibi)

        self.data = self.data[len(ibis):]

        return ibis

    def _get_file_data(self):

        for (fp, ts) in self.timestamps:
            with open(fp) as f:
                f.readline() # timestamp
                f.readline() # frequency

            self.num_used_files += 1
            self.rounds_per_file.append(0)

            new_data = [self.pl(line) or line in f]

            if len(new_data) < self.window:
                raise ValueError(
                    'File must have at least as many lines and hertz * seconds.')

            yield new_data

    def cols(self):

        return self.window

    def rows(self):

        return self.num_rounds
