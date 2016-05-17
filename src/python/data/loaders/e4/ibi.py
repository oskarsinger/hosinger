from data.loaders import AbstractDataLoader
from global_utils import file_io as fio
from math import ceil, floor

import os

import numpy as np

class IBILoader(AbstractDataLoader):

    def __init__(self, dir_path, filename, seconds, reader):

        self.dir_path = dir_path
        self.filename = filename
        self.reader = reader
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
        self.data = None

    def get_data(self):

        if self.data is None:
            self._set_data()

        return np.copy(self.data)

    def _set_data(self):

        data = []

        for (fp, ts) in self.timestamps:
            print fp
            with open(fp) as f:
                # Clear out timestamp on first line
                f.readline()

                # Populate data list with remaining lines
                file_data = [self.reader(line) for line in f]
                print len(file_data)

                # Extract event-based representation
                data.extend(self._get_rows(file_data))

        self.data = np.array(data)

    def _get_rows(self, data):
        
        # Last second in which an event occurs
        end = int(ceil(data[-1][0]))

        # Initialize iteration variables
        rows = []
        i = 1
        
        # Iterate until entire window is after all recorded events
        while (i - 1) * self.seconds < end:

            (row, data) = self._get_row(data, i)

            # Update iteration variables
            rows.append(row)
            i += 1

        return rows

    def _get_row(self, data, i):

        events = {}

        for j, (time, value) in enumerate(data):

            # If time of measurement is outside window
            if time >= i * self.seconds:
                # Truncate measurements already seen
                data = data[j:]

                break

            # The second in which event occurred
            second = int(floor(time))

            if second in events:
                events[second].append(value)
            else:
                events[second] = [value]

        # Enumerate the seconds of interest
        window = [i * k for k in range(self.seconds)]

        # Give statistic of events occuring in these seconds
        row = [0 if s not in events else len(events[s])
               for s in window]

        return (row, data)

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
