from data.loaders import AbstractDataLoader
from global_utils import file_io as fio
from global_utils.misc import get_list_mod as get_lm

import os

import numpy as np

class FixedRateLoader(AbstractDataLoader):

    def __init__(self, dir_path, filename, seconds, reader, hertz):

        self.dir_path = dir_path
        self.filename = filename
        self.reader = reader
        self.seconds = seconds
        self.hertz = hertz

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

    def get_data(self):

        if self.data is None:
            self._set_data()

        return np.copy(self.data)

    def _set_data(self):

        data = []

        for (fp, ts) in self.timestamps:
            with open(fp) as f:
                # Clear out timestamp on first line
                f.readline()
                # Clear out frequency on second line
                f.readline()

                # Populate data list with remaining lines
                file_data = [self.reader(line) for line in f]

                # Attach modded list to full data set
                data.extend(self._get_rows(file_data))

        self.data = np.array(data)

    def _get_rows(self, data):

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
            'rounds_per_file': self.rounds_per_file,
            'process_line': self.pl,
            'data': self.data}

    def cols(self):

        return self.window

    def rows(self):

        return self.data.shape[0] if data is not None else 0
