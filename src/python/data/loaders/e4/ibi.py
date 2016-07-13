from data.loaders import AbstractDataLoader
from data.missing import MissingData
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
        self.num_sessions = len(self._get_hdf5_repo())

        self.data = None
        self.num_rounds = 0
        self.current_time = None

    def get_data(self):

        if self.online:
            self._refill_data()
        elif self.data is None:
            self._set_data()

        self.num_rounds += 1

        return self.data.astype(float)

    def _refill_data(self):

        sessions = self._get_hdf5_repo()
        index = self.num_rounds % len(sessions)
        sorted_sessions = sorted(
                sessions.items(), key=lambda x: x[0])
        (key, session) = sorted_sessions[index]

        self.data = np.copy(self._get_rows(key, session))

    def _set_data(self):

        data = None
        repo = self._get_hdf5_repo()

        for (ts, session) in repo.items():
            if data is None:
                data = self._get_rows(session)
            else:
                data = np.vstack(
                    [data, self._get_rows(session)])

        self.data = np.copy(data)

    def _get_rows(self, key, session):

        # Use key to check if there is a significant gap between measurements
        # If so, serve a MissingData object with number of MissingData rounds
        # Otherwise, server the data itself
        # Either way, update current_time accordingly

        # Get dataset associated with relevant sensor
        hdf5_dataset = session[self.sensor]

        # Populate entry list with entries of hdf5 dataset
        read_data = self.reader(hdf5_dataset)

        # Return the extracted windows of the data
        return self._get_event_windows(read_data)

    def _get_event_windows(self, data):
        
        # Last second in which an event occurs
        end = int(ceil(data[-1,0]))

        # Initialize iteration variables
        rows = None
        i = 1
        
        # Iterate until entire window is after all recorded events
        while (i - 1) * self.seconds < end:

            row = self._get_row(data, i)

            # Update iteration variables
            if rows is None:
                rows = row
            else:
                rows = np.vstack([rows, row])

            i += 1

        return rows

    def _get_row(self, data, i):

        row = np.zeros(self.seconds)[:,np.newaxis].T
        begin = (i - 1) * self.seconds
        end = begin + self.seconds
        time = data[:,0]

        for i in xrange(self.seconds):
            relevant = np.logical_and(time >= begin, time < end)
            row[0,i] = np.count_nonzero(relevant)

        return row

    def _get_hdf5_repo(self):

        return h5py.File(self.hdf5_path, 'r')[self.subject]

    def cols(self):

        return self.seconds

    def rows(self):

        rows = 0

        if self.online:
            rows = self.num_rounds
        elif self.data is not None:
            rows = self.data.shape[0]

        return rows

    def finished(self):

        finished = None

        if self.online:
            finished = self.num_rounds > self.num_sessions
        else:
            finished = self.num_rounds > 1

        return finished

    def name(self):

        return self.sensor

    def refresh(self):

        self.data = None
        self.num_rounds = 0

    def get_status(self):

        return {
            'hdf5_path': self.hdf5_path,
            'subject': self.subject,
            'sensor': self.sensor,
            'seconds': self.seconds,
            'num_rounds': self.num_rounds,
            'num_sessions': self.num_sessions,
            'reader': self.reader,
            'data': self.data,
            'online': self.online}
