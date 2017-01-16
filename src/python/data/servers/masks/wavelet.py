import h5py
import os

import numpy as np
import wavelets.dtcwt as dtcwt

from drrobert.file_io import get_timestamped as get_ts
from wavelets.dtcwt.utils import get_partial_reconstruction as get_pr
from linal.utils.misc import get_array_mod

class WaveletMask:

    def __init__(self,
        ds,
        period,
        max_freqs,
        save_load_path,
        load=False,
        save=False,
        get_pr=None):

        # For now, assume server is batch-style
        self.ds = ds
        self.get_wavelet_transform = get_wavelet_transform
        self.period = period
        self.hertz = self.ds.get_status()['data_loader'].get_status()['hertz']
        self.max_freqs = max_freqs
        self.save_load_path = save_load_path
        self.get_pr = pr
        self.load = load
        self.save = save

        self.window = int(self.period * self.hertz)
        self.num_freqs = min([
            int(log(self.window, 2)) - 1,
            self.max_freqs])
        self.biorthogonal = wdtcwt.utils.get_wavelet_basis(
            'near_sym_b')
        self.qshift = wdtcwt.utils.get_wavelet_basis(
            'qshift_b')
        
        hdf5_repo = None

        if self.load:
            hdf5_repo = h5py.File(
                self.save_load_path, 'r')
        else:
            name = get_ts('_'.join([
                'f',
                str(self.hertz),
                'p',
                str(self.period),
                'mf',
                str(self.max_freqs)])) + '.hdf5'

            self.save_load_path = os.path.join(
                save_load_path, name)

            hdf5_repo = h5py.File(
                self.save_load_path, 'w')

        self.hdf5_repo = hdf5_repo

    def get_data(self):

        wavelets = []

        if self.load:
            for i in xrange(len(self.hdf5_repo)):
                group = self.hdf5_repo[str(i)]
                num_Yh = len(group) - 1
                Yh = [group['Yh_' + str(j)] for j in xrange(num_Yh)]
                Yl = group['Yl']

                wavelets.insert((Yh, Yl))
        else:
            data = self.ds.get_data()
            num_rows = int(float(data.shape[0]) / self.window)
            reshaped = np.reshape(
                get_array_mod(data, self.window),
                (num_rows, self.window))

            for i in xrange(num_rows):
                key = str(i)
                (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                    reshaped[i,:][:,np.newaxis],
                    num_freqs - 1,
                    self.biorthogonal,
                    self.qshift)

                wavelets.insert((Yh, Yl))

                if self.save:
                    self.hdf5_repo.create_group(key)
                    group = self.hdf5_repo[key]

                    for (j, freq) in enumerate(Yh):
                        group[key].create_dataset(
                            'Yh_' + str(j), freq)

                    group[key].create_dataset(
                        'Yl', freq)

        if self.pr:
            wavelets = [get_pr(Yh, Yl) for (Yh, Yl) in wavelets]

        return wavelets

    def cols(self):

        return self.ds.cols()

    def rows(self):

        return self.ds.rows()

    def refresh(self):

        self.ds.refresh()

    def get_status(self):
        
        new_status = {
            'ds': self.ds,
            'period': self.period,
            'hertz': self.hertz,
            'max_freqs': self.max_freqs,
            'save_load_path': self.save_load_path,
            'pr': self.get_pr,
            'load': self.load,
            'save': self.save,
            'window': self.window,
            'hdf5_repo': self.hdf5_repo}

        for (k, v) in self.ds.get_status():
            new_status[k] = v

        return new_status
