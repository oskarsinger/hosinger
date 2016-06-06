from data.loaders import AbstractDataLoader
from math import ceil, floor

import os
import h5py

import numpy as np

class IBILoader(AbstractDataLoader):

    def __init__(self,
        hdf5_path, subject, sensor, seconds, reader,
        online=False):

        self.hdf5_path = hdf5_path
        self.subject = subject
        self.sensor = sensor
        self.reader = reader
        self.seconds = seconds
        self.online = online

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

        sessions = self._get_hdf5_repo()
        index = self.num_rounds % len(sessions)
        hdf5_dataset = sessions.values()[index]

        self.data = np.array(self._get_file_rows(hdf5_dataset))

    def _set_data(self):

        data = []
        repo = self._get_hdf5_repo()

        for (ts, sensors) in repo.items():
            hdf5_dataset = sensors[self.sensor]
            data.extend(self._get_rows(hdf5_dataset))

        self.data = np.array(data)

    def _get_rows(self, hdf5_dataset):

        # Populate entry list with entries of hdf5 dataset
        entries = [self.reader(entry) 
                   for entry in hdf5_dataset]

        # Return the extracted windows of the data
        return self._get_event_windows(entries)

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

    def _get_hdf5_repo(self):

        return h5py.File(self.hdf5_path, 'r')[self.subject]

    def get_status(self):

        return {
            'hdf5_path': self.hdf5_path,
            'subject': self.subject,
            'sensor': self.sensor,
            'seconds': self.seconds,
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
