from data.loaders import AbstractDataLoader
from data.pseudodata import MissingData
from linal.utils import get_array_mod
from math import ceil
from datetime import datetime as DT
from time import mktime

import os
import h5py

import numpy as np

class FixedRateLoader(AbstractDataLoader):

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

        # Set the sampling frequency
        repo = self._get_hdf5_repo()
        self.start_times = [self._get_datetime_from_key(k)
                            for k in sorted(repo.keys())]
        self.num_sessions = len(self.start_times)
        dataset = repo.values()[0][self.sensor]
        key = [k for k in dataset.attrs.keys() if 'hz' in k][0]
        self.hertz = dataset.attrs[key]
        self.window = 1

        if self.seconds is not None: 
            self.window = int(self.hertz * self.seconds)
        else:
            self.seconds = 1.0 / self.hertz

        self.data = None
        self.on_deck_data = None
        self.num_rounds = 0
        self.num_real_data = 0
        self.current_time = None

    def get_data(self):

        if self.online:
            self._refill_data()
        elif self.data is None:
            self._set_data()

        batch = self.data

        if not isinstance(self.data, MissingData):
            batch = np.copy(self.data.astype(float))
            self.num_real_data += 1

        self.num_rounds += 1

        return batch

    def _refill_data(self):

        sessions = self._get_hdf5_repo().items()
        index = self.num_real_data % len(sessions)
        sorted_sessions = sorted(sessions, key=lambda x: x[0])
        (key, session) = sorted_sessions[index]

        if self.on_deck_data is not None:
            self.data = self.on_deck_data
            self.on_deck_data = None
        else:
            self.data = self._get_rows(key, session)

    def _set_data(self):

        data = None
        repo = self._get_hdf5_repo()

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

        self.data = data

    def _get_rows(self, key, session):

        # Get dataset associated with relevant sensor
        hdf5_dataset = session[self.sensor]

        # Populate entry list with entries of hdf5 dataset
        read_data = self.reader(hdf5_dataset)

        # Get the extracted windows of the data
        data = self._get_windows(read_data)

        # Get difference between self.current_time and session's start time
        time_diff = self._get_time_difference(key)

        self.current_time += time_diff
        self.current_time += data.shape[0] * self.seconds

        if time_diff > 0:
            self.on_deck_data = data
            num_missing_rows = int(ceil(time_diff/self.seconds))
            data = MissingData(num_missing_rows) 

        return data

    def _get_time_difference(self, key):

        dt = self._get_datetime_from_key(key)
        uts = mktime(dt.timetuple())

        if self.current_time is None:
            self.current_time = uts

        time_diff = uts - self.current_time

        return time_diff

    def _get_datetime_from_key(self, key):

        (date_str, time_str) = key.split('_')[1].split('-')
        (year, month, day) = [int(date_str[2*i:2*(i+1)])
                              for i in range(3)]
        year += 2000
        (hour, minute, second) = [int(time_str[2*i:2*(i+1)])
                                  for i in range(3)]
        dt = DT(year, month, day, hour, minute, second)

        return dt

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

        # TODO: rewrite this; its wrong
        rows = 0

        if self.online:
            rows = self.num_rounds
        elif self.data is not None:
            rows = self.data.shape[0]

        return rows

    def finished(self):

        finished = None

        if self.online:
            finished = self.num_real_data >= self.num_sessions
        else:
            finished = self.num_real_data >= 1

        return finished

    def name(self):

        return self.sensor

    def refresh(self):

        self.data = None
        self.num_rounds = 0
        self.num_real_data = 0

    def get_status(self):

        return {
            'hdf5_path': self.hdf5_path,
            'subject': self.subject,
            'sensor': self.sensor,
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
            'num_rounds': self.num_rounds,
            'num_real_data': self.num_real_data,
            'num_sessions': self.num_sessions,
            'start_times': self.start_times,
            'reader': self.reader,
            'data': self.data,
            'online': self.online}
