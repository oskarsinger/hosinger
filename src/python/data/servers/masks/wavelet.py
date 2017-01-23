import h5py
import os

import numpy as np
import wavelets.dtcwt as dtcwt

from drrobert.file_io import get_timestamped as get_ts
from wavelets.dtcwt.utils import get_padded_wavelets as get_pw
from wavelets.dtcwt.oned import get_partial_reconstructions as get_pr
from linal.utils.misc import get_array_mod

class DTCWTMask:

    def __init__(self,
        ds,
        save_load_path,
        period=3600,
        max_freqs=7,
        padded=True,
        pr=False,
        serve_one_period=True,
        load=False,
        save=False):

        # For now, assume server is batch-style
        self.ds = ds
        self.period = period
        self.hertz = self.ds.get_status()['data_loader'].get_status()['hertz']
        self.max_freqs = max_freqs
        self.save_load_path = save_load_path
        self.padded = padded
        self.pr = pr
        self.serve_one_period = serve_one_period
        self.load = load
        self.save = save

        self.window = int(self.period * self.hertz)
        self.num_freqs = min([
            int(log(self.window, 2)) - 1,
            self.max_freqs])
        self.num_rounds = 0
        self.biorthogonal = wdtcwt.utils.get_wavelet_basis(
            'near_sym_b')
        self.qshift = wdtcwt.utils.get_wavelet_basis(
            'qshift_b')
        self.data = None

        if not self.load:
            data = self.ds.get_data()
            num_rows = int(float(data.shape[0]) / self.window)
            reshaped = np.reshape(
                get_array_mod(data, self.window),
                (num_rows, self.window))
            self.data = reshaped
        
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

        wavelets = None

        if self.serve_one_period:
            wavelets = self._get_one_period(self.num_rounds)

            self.num_rounds += 1
        else:
            num_rounds = len(self.hdf5_repo) \
                if self.load else \
                self.data.shape[0]
            wavelets = [self._get_one_period(i)
                        for i in xrange(num_rounds)]

        return wavelets

    def _get_one_period(self, i):

        wavelets = None

        if self.load:
            group = self.hdf5_repo[str(i)]
            num_Yh = len(group) - 1
            Yh = [np.array(group['Yh_' + str(j)]) 
                  for j in xrange(num_Yh)]
            Yl = np.array(group['Yl'])
            wavelets = (Yh, Yl)
        else:
            key = str(i)
            (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                self.data[i,:][:,np.newaxis],
                num_freqs - 1,
                self.biorthogonal,
                self.qshift)
            wavelets = (Yh, Yl)

            if self.save:
                self.hdf5_repo.create_group(key)
                group = self.hdf5_repo[key]

                for (j, freq) in enumerate(Yh):
                    group[key].create_dataset(
                        'Yh_' + str(j), freq)

                group[key].create_dataset(
                    'Yl', freq)

        if self.padded:
            wavelets = get_pw(Yh, Yl)
        elif self.pr:
            wavelets = get_pr(Yh, Yl, self.biorthogonal, self.qshift)

        return wavelets

    def _get_all_periods(self):

        wavelets = []

        if self.load:
            for i in xrange(len(self.hdf5_repo)):
                group = self.hdf5_repo[str(i)]
                num_Yh = len(group) - 1
                Yh = [np.array(group['Yh_' + str(j)]) 
                      for j in xrange(num_Yh)]
                Yl = np.array(group['Yl'])

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

        if self.padded:
            wavelets = [get_pw(Yh, Yl) for (Yh, Yl) in wavelets]
        elif self.pr:
            wavelets = [get_pr(Yh, Yl, self.biorthogonal, self.qshift)
                        for (Yh, Yl) in wavelets]

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
            'padded': self.padded,
            'pr': self.pr,
            'load': self.load,
            'save': self.save,
            'window': self.window,
            'hdf5_repo': self.hdf5_repo}

        for (k, v) in self.ds.get_status():
            if k not in new_status:
                new_status[k] = v

        return new_status
