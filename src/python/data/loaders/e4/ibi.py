from data.loaders import AbstractDataLoader
from drrobert import file_io as fio
from math import ceil, floor
from random import choice

import os

import numpy as np

class IBILoader(AbstractDataLoader):

    def __init__(self,
        dir_path, filename, seconds, reader,
        online=False):

        self.dir_path = dir_path
        self.filename = filename
        self.reader = reader
        self.seconds = seconds
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

            # Populate data list with remaining lines
            file_data = [self.reader(line) for line in f]

            # Extract event-based representation
            data = self._get_event_windows(file_data)

        return data

    def _get_event_windows(self, data):
        
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
            'reader': self.reader,
            'data': self.data,
            'online': self.online}

    def cols(self):

        return self.seconds

    def rows(self):

        rows = 0

        if self.online:
            rows = self.num_rounds
        elif self.data is not None:
            rows = self.data.shape[0]

        return rows
