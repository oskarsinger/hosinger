from data.loaders import AbstractDataLoader
from linal.utils import get_array_mod

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

        # Set the sampling frequency
        dataset = self._get_hdf5_repo().values()[0][self.sensor]
        key = [k for k in dataset.attrs.keys() if 'hz' in k][0]
        self.hertz = dataset.attrs[key]

        self.window = int(self.hertz * self.seconds)
        self.data = None
        self.num_rounds = 0

    def get_data(self):

        if self.online:
            self._refill_data()

            self.num_rounds += 1
        elif self.data is None:
            self._set_data()

        return np.copy(self.data).astype(float)

    def _refill_data(self):

        sessions = self._get_hdf5_repo()
        index = self.num_rounds % len(sessions)
        session = sessions.values()[index]

        self.data = np.copy(self._get_rows(session))

    def _set_data(self):

        data = None
        repo = self._get_hdf5_repo()

        for (ts, session) in repo.items():
            if data is None:
                data = self._get_rows(session)
            else:
                data = np.concatenate(
                    (data,self._get_rows(session)))

        self.data = np.copy(data)

    def _get_rows(self, session):

        # Get dataset associated with relevant sensor
        hdf5_dataset = session[self.sensor][self.sensor.lower()]

        # Populate entry list with entries of hdf5 dataset
        read_data = self.reader(hdf5_dataset)

        # Return the extracted windows of the data
        return self._get_windows(read_data)

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

    def get_status(self):

        return {
            'hdf5_path': self.hdf5_path,
            'subject': self.subject,
            'sensor': self.sensor,
            'hertz': self.hertz,
            'seconds': self.seconds,
            'window': self.window,
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
