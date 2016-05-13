from data.loaders import AbstractDataLoader
from global_utils import file_io as fio, get_list_mod as get_lm

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
        self.window = int(self.hertz * self.seconds)
        self.data = None

    def get_datum(self):

        if self.data is None:
            self.data = self._get_file_data()

        # Remove that window from data queue
        self.data = self.data[self.window:]

        return np.array(datum)

    def _get_file_data(self):

        data = []

        for (fp, ts) in timestamps:

            with open(fp) as f:
                # Clear out timestamp on first line
                f.readline()
                # Clear out frequency on second line
                f.readline()

                # Populate data list with remaining lines
                file_data = [self.pl(line) for line in f]
                data.append(get_lm(file_data))

        return new_data

    def get_status(self):

        return {
            'dir_path': self.dir_path,
            'filename': self.filename,
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'timestamps': self.filepaths,
            'num_rounds': self.num_rounds,
            'rounds_per_file': self.rounds_per_file,
            'process_line': self.pl,
            'data': self.data}

    def cols(self):

        return self.window

    def rows(self):

        return self.data.shape[0] if data is not None else 0
