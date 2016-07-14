from data.loaders import AbstractDataLoader
from data.missing import MissingData
from linal.utils import get_array_mod
from math import ceil
from datetime import datetime as DT

import os
import h5py

import numpy as np

class FixedRateLoader(AbstractDataLoader):

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

        # Set the sampling frequency
        dataset = self._get_hdf5_repo().values()[0][self.sensor]
        key = [k for k in dataset.attrs.keys() if 'hz' in k][0]
        self.hertz = dataset.attrs[key]

        self.window = int(self.hertz * self.seconds)
        self.data = None
        self.num_rounds = 0
        self.current_time = None

    def get_data(self):

        if self.online:
            self._refill_data()
        elif self.data is None:
            self._set_data()

        self.num_rounds += 1

        batch = self.data

        print type(self.data)
        if type(self.data) is not MissingData:
            print self.data
            batch = np.copy(self.data.astype(float))

        return batch

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

        self.data = data

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
            data = self._get_windows(read_data)

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

    def _get_windows(self, data):

        rows = None

        # If there are enough measurements for desired window
        if len(data) >= self.window:
            # Prepare data to be merged into full list
            modded = get_array_mod(data, self.window)
            length = modded.shape[0] / self.window
            rows = modded.reshape((length, self.window))
        else:
            raise ValueError(
                'File has less than hertz * seconds lines.')

        return rows

    def _get_hdf5_repo(self):

        return h5py.File(self.hdf5_path, 'r')[self.subject]

    def cols(self):

        return self.window

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
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'num_rounds': self.num_rounds,
            'num_sessions': self.num_sessions,
            'reader': self.reader,
            'data': self.data,
            'online': self.online}
