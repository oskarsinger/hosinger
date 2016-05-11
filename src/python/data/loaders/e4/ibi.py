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
        self.rounds_per_file = []
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

        return len(ibis)

    def _get_file_data(self):

        fp = self.timestamps[len(self.rounds_per_file)][0]
        new_data = None

        with open(fp) as f:
            # Clear out timestamp on first line
            f.readline()

            # Populate data list with remaining lines
            new_data = [self.pl(line) for line in f]

        self.rounds_per_file.append(0)

        if len(new_data) < self.window:
            raise ValueError(
                'File must have at least hertz * seconds lines.')

        return new_data

    def get_status(self):

        return {
            'dir_path': self.dir_path,
            'filename': self.filename,
            'seconds': self.seconds,
            'timestamps': self.filepaths,
            'num_rounds': self.num_rounds,
            'rounds_per_file': self.rounds_per_file,
            'process_line': self.pl,
            'data': self.data}

    def cols(self):

        return 1

    def rows(self):

        return self.num_rounds
