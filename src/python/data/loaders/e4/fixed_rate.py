from data.loaders import AbstractDataLoader
from drrobert.misc import get_list_mod as get_lm

import os
import h5py

import numpy as np

class FixedRateLoader(AbstractDataLoader):

    def __init__(self,
        hdf5_path, subject, sensor, seconds, reader, hertz,
        online=False):

        self.hdf5_path = hdf5_path
        self.subject = subject
        self.sensor = sensor
        self.reader = reader
        self.seconds = seconds
        self.hertz = hertz
        self.online = online

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
        return self._get_windows(file_data)

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
