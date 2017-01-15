import h5py
import os

import numpy as np

from drrobert.file_io import get_timestamped as get_ts

class WaveletMask:

    def __init__(self,
        ds,
        get_wavelet_transform,
        period,
        frequency,
        max_freqs,
        save_load_path,
        load=False):

        # For now, assume server is batch-style
        self.ds = ds
        self.get_wavelet_transform = get_wavelet_transform
        self.period = period
        self.frequency = frequency
        self.max_freqs = max_freqs
        self.save_load_path = save_load_path
        self.load = load

        self.window = int(self.period * self.frequency)
        
        hdf5_repo = None

        if self.load:
            hdf5_repo = h5py.File(
                self.save_load_path, 'r')
        else:
            name = get_ts('_'.join([
                'f',
                str(frequency),
                'p',
                str(period),
                'mf',
                str(max_freqs)])) + '.hdf5'

            self.save_load_path = os.path.join(
                save_load_path, name)

            hdf5_repo = h5py.File(
                self.save_load_path, 'w')

        self.hdf5_repo = hdf5_repo

    def get_data(self):

        wavelets = None

        if self.load:
            # TODO: fill this in
            wavelets = self.hdf5_repo
        else:
            # Probably assume column vector for now
            # TODO: fill this in
            data = self.ds.get_data()
            wavelets = 'Poop'

        return wavelets

    def save(self, period):

        # TODO: fill this in
