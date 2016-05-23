from data.loaders import AbstractDataLoader
from drrobert import file_io as fio
from drrobert.misc import get_list_mod as get_lm
from random import choice

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

    def __init__(self,
        dir_path, filename, seconds, reader, hertz,
        online=False):

        self.dir_path = dir_path
        self.filename = filename
        self.reader = reader
        self.seconds = seconds
        self.hertz = hertz
        self.online = online
        self.random = random

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
        self.window = int(self.hertz * self.seconds)
        self.data = None
        self.num_rounds = 0

    def get_data(self):

        if self.online:
            self._refill_data()

            self.num_rounds += 1
        elif self.data is None:
            self._set_data()

        return np.copy(self.data)

    def _refill_data(self):

        index = self.num_rounds % len(self.timestamps)
        fp = self.timestamps[index][0]

        self.data = np.array(self._get_file_rows(fp))
        

    def _set_data(self):

        data = []

        for (fp, ts) in self.timestamps:
            data.extend(self._get_file_rows(fp))

        self.data = np.array(data)

    def _get_file_rows(self, fp):

        data = []

        with open(fp) as f:
            # Clear out timestamp on first line
            f.readline()
            # Clear out frequency on second line
            f.readline()

            # Populate data list with remaining lines
            file_data = [self.reader(line) for line in f]

            # Attach modded list to full data set
            data = self._get_windows(file_data)

        return data

    def _get_windows(self, data):

        rows = None

        # If there are enough measurements for desired window
        if len(data) >= self.window:
            # Prepare data to be merged into full list
            modded = get_lm(data, self.window)
            rows = [modded[i*self.window:(i+1)*self.window]
                    for i in xrange(len(modded)/self.window)]
        else:
            raise ValueError(
                'File ' + fp + ' has less than hertz * seconds lines.')

        return rows

    def get_status(self):

        return {
            'dir_path': self.dir_path,
            'filename': self.filename,
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'timestamps': self.filepaths,
            'num_rounds': self.num_rounds,
            'reader': self.reader,
            'data': self.data,
            'online': self.online}

    def cols(self):

        return self.window

    def rows(self):

        rows = 0

        if self.online:
            rows = self.num_rounds
        elif self.data is not None:
            rows = self.data.shape[0]

        return rows
