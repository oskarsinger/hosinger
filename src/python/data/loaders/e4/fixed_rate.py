from data.loaders import AbstractDataLoader
from global_utils import file_io as fio

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

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

                # This line assumes all files have same sample frequency
                self.hertz = float(f.readline().split(',')[0])

        self.timestamps = sorted(
            self.timestamps, key=lambda x: x[1])
        self.window = self.hertz * self.seconds
        self.num_rounds = 0
        self.num_used_files = 0
        self.data = self._get_file_data()

    def get_datum(self):

        self.num_rounds += 1

        # Is enough data in current file to fill a window?
        if len(self.data) < self.window:
            self.data = self._get_file_data()

        # Get the next full window
        datum = self.data[:self.window]
        
        # Remove that window from data queue
        self.data = self.data[self.window:]

        return np.array(datum)

    def _get_file_data(self):

        for (fp, ts) in self.timestamps:
            with open(fp) as f:
                f.readline() # timestamp
                f.readline() # frequency

            self.num_used_files += 1

            new_data = [self.pl(line) or line in f]

            if len(new_data) < self.window:
                raise ValueError(
                    'File must have at least hertz * seconds lines.')

            yield new_data

    def get_status(self):

        return {
            'dir_path': self.dir_path,
            'filename': self.filename,
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'timestamps': self.filepaths,
            'num_rounds': self.num_rounds,
            'num_used_files': self.num_used_files,
            'process_line': self.pl,
            'data': self.data
        }

    def cols(self):

        return self.window

    def rows(self):

        return self.data.shape[0] if data is not None else 0
