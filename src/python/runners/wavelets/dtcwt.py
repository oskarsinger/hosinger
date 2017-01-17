import os
import json
import h5py

import numpy as np
import data.loaders.shortcuts as dls
import wavelets.dtcwt as wdtcwt
import utils as rmu

from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan
from wavelets import dtcwt
from multiprocessing import Pool
from math import log

class MVDTCWTRunner:

    def __init__(self, 
        data_path=None,
        period=24*3600,
        subperiod=3600,
        max_freqs=10,
        dataset='e4',
        save_load_dir=None, 
        save=False,
        load=False):

        self.data_path = data_path
        self.dataset = dataset
        self.period = period
        self.subperiod = subperiod
        self.max_freqs = max_freqs
        self.num_sps = self.period / self.subperiod

        self.biorthogonal = wdtcwt.utils.get_wavelet_basis(
            'near_sym_b')
        self.qshift = wdtcwt.utils.get_wavelet_basis(
            'qshift_b')

        self._init_dirs(
            save_load_dir,
            load,
            save)
        self._init_server_stuff()

        self.wavelets = {s : [] for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

    def _init_server_stuff(self):

        print 'Initializing servers'

        loaders = None

        if self.dataset == 'ats':
            loaders = dls.get_ats_loaders_all_subjects(
                self.data_path)
        elif self.dataset == 'e4':
            loaders = dls.get_e4_loaders_all_subjects(
                self.data_path, None, False)
        elif self.dataset == 'atr':
            loaders = dls.get_atr_loaders()
        elif self.dataset == 'gr':
            ps = [1] * 2
            hertzes = [1.0/60] * 2
            n = 60 * 24 * 8
            loaders = {'e' + str(i): dls.get_FPGL(n, ps, hertzes)
                       for i in xrange(2)}
        else:
            raise ValueError('Argument to dataset parameter not valid.')

        self.servers = {}

        for (s, dl_list) in loaders.items():
            s = s[-2:]

            # This is to ensure that all subjects have sufficient data
            # Only necessary if we have to recompute wavelets
            try:
                if not self.load:
                    data = [dl.get_data() for dl in dl_list]

                self.servers[s] = [BS(dl) for dl in dl_list]
            except Exception, e:
                print 'Could not load data for subject', s
                print e
        
        (self.rates, self.names) = unzip(
            [(dl.get_status()['hertz'], dl.name())
             for dl in loaders.items()[0][1]])

        if self.dataset == 'gr':
            self.names = [n + str(i) 
                          for (i, n) in enumerate(self.names)]
                    
        self.subjects = self.servers.keys()
        
        # Avoids loading data if we don't need to
        if not self.load:
            server_np = lambda ds, r: ds.rows() / (r * self.period)
            subject_np = lambda s: min(
                [server_np(ds, r) 
                 for (ds, r) in zip(self.servers[s], self.rates)])
            self.num_periods = {s : int(subject_np(s))
                                for s in self.subjects}
            path = os.path.join(
                self.save_load_dir,
                'num_periods.json')
            np_json = json.dumps(self.num_periods)

            with open(path, 'w') as f:
                f.write(np_json)
        else:
            path = os.path.join(
                self.save_load_dir,
                'num_periods.json')

            with open(path) as f:
                l = f.readline().strip()
                self.num_periods = json.loads(l)

            info = self.save_load_dir.split('_')
            self.subperiod = int(info[-1])
            self.period = int(info[-3])
            self.num_sps = self.period / self.subperiod

        self.num_views = len(self.servers.items()[0][1])

    def _init_dirs(self,
        save_load_dir,
        load,
        save):

        self.load = load
        self.save = save

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('_'.join([
                'MVDTCWTR',
                'period',
                str(self.period),
                'subperiod',
                str(self.subperiod)]))

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        hdf5_path = os.path.join(
            self.save_load_dir,
            'wavelets')
        
        self.hdf5_repo = h5py.File(
            hdf5_path,
            'w' if save else 'r')

    def _compute(self):

        for s in self.subjects:

            print 'Computing wavelet transforms for subject', s

            iterable = enumerate(zip(
                self.rates, self.servers[s]))

            for (v, (r, ds)) in iterable:
                window = int(r * self.period)
                sp_window = int(self.subperiod * r)
                threshold = self.names[v] == 'TEMP'
                data = ds.get_data()

                for p in xrange(self.num_periods[s]):
                    data_p = data[p * window: (p+1) * window]

                    for sp in xrange(self.num_sps):
                        begin = sp * sp_window
                        end = begin + sp_window
                        data_sp = data_p[begin:end]

                        if threshold:
                            data_sp[data_sp > 40] = 40

                        num_freqs = min([
                            int(log(data_sp.shape[0], 2)) - 1,
                            self.max_freqs])
                        (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                            data_sp, 
                            num_freqs - 1,
                            self.biorthogonal,
                            self.qshift)

                        if num_freqs > 7:
                            Yhs = Yhs[-6:]

                        self._save(
                            Yl,
                            Yh,
                            s,
                            v,
                            p,
                            sp)

    def _load(self):

        print 'Loading wavelets'

        self.wavelets = rmu.get_wavelet_storage(
            self.num_views,
            self.num_sps,
            self.num_periods,
            self.subjects)

        for (s, s_group) in self.hdf5_repo.items():
            for (v_str, v_group) in s_group:
                v = int(v_str)

                for (p_str, p_group) in v_group:
                    p = int(p_str)

                    for (sp_str, sp_group) in p_group:
                        sp = int(sp_str)
                        num_yh = len(sp_group) - 1
                        Yh = [sp_group['Yh_' + str(i)]
                              for i in xrange(num_yh)]

                        self.wavelets[s][p][sp][v][0] = (Yh, sp_group['Yl'])

    def _save(self, Yl, Yh, s, v, p, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_str = str(v)

        if v_str not in s_group:
            s_group.create_group(v_str)

        v_group = s_group[v_str]
        p_str = str(p)

        if p_str not in v_group:
            v_group.create_group(p_str)

        p_group = v_group[p_str]
        sp_str = str(sp)

        p_group.create_group(sp_str)

        sp_group = p_group[sp_str]

        sp_group.create_dataset('Yl', data=Yl)

        for (i, yh) in Yh:
            sp_group.create_dataset(
                'Yh_' + str(i), data=yh)
