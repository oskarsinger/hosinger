from data.pseudodata import MissingData
from math import ceil, floor
from datetime import datetime as DT

import os
import h5py

import numpy as np

class IBILoader:

    def __init__(self,
        hdf5_path, subject, sensor, reader,
        seconds=None,
        online=False):

        self.hdf5_path = hdf5_path
        self.subject = subject
        self.sensor = sensor
        self.reader = reader
        self.seconds = seconds
        self.online = online
        self.num_sessions = len(self._get_hdf5_repo())

        # TODO: figure out what to do when seconds == None

        self.data = None
        self.num_rounds = 0
        self.current_time = None

    def get_data(self):

        if self.online:
            self._refill_data()
        elif self.data is None:
            self._set_data()

        batch = self.data

        if not isinstance(self.data, MissingData):
            batch = np.copy(self.data.astype(float))

        return batch

    def _refill_data(self):

        sessions = self._get_hdf5_repo()
        index = self.num_rounds % len(sessions)
        sorted_sessions = sorted(
                sessions.items(), key=lambda x: x[0])
        (key, session) = sorted_sessions[index]

        self.data = self._get_rows(key, session)

    def _set_data(self):

        data = None
        repo = self._get_hdf5_repo()

        # TODO: fix this later
        """
        for (ts, session) in repo.items():
            new_data = self._get_rows(ts, session)

            if data is None:
                data = new_data
            elif self.on_deck_data is not None:
                data = np.vstack(
                    [data, self.on_deck_data])

                self.on_deck_data = None
            else:
                    data = np.vstack(
                        [data, new_data])
        """

        self.data = np.copy(data)

    def _get_rows(self, key, session):

        time_diff = self._get_time_difference(key)
        data = None

        if time_diff >= 1:
            num_missing_rows = int(ceil(time_diff/self.seconds))
            data = MissingData(num_missing_rows) 
        else:
            # Get dataset associated with relevant sensor
            hdf5_dataset = session[self.sensor]

            # Populate entry list with entries of hdf5 dataset
            read_data = self.reader(hdf5_dataset)

            # Get the extracted windows of the data
            data = self._get_event_windows(read_data)

        return data

    def _get_time_difference(self, key):

        (date_str, time_str) = key.split('_')[1].split('-')
        (year, month, day) = [int(date_str[2*i:2*(i+1)])
                              for i in range(3)]
        (hour, minute, second) = [int(time_str[2*i:2*(i+1)])
                                  for i in range(3)]
        dt = DT(year, month, day, hour, minute, second)
        uts = (dt - DT.utcfromtimestamp(0)).total_seconds()
        time_diff = 0

        if self.current_time is None:
            self.current_time = uts
        else:
            time_diff = uts - self.current_time
            self.current_time += time_diff

        return time_diff

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
            finished = self.num_rounds >= self.num_sessions
        else:
            finished = self.num_rounds >= 1

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
