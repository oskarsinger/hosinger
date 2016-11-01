import os
import json

import numpy as np
import data.loaders.e4.shortcuts as dles
import data.loaders.at.shortcuts as dlas
import data.loaders.synthetic.shortcuts as dlss
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
        dataset='e4',
        save_load_dir=None, 
        save=False,
        load=False):

        self.data_path = data_path
        self.dataset = dataset
        self.period = period
        self.subperiod = subperiod
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

        if self.dataset == 'at':
            loaders = dlas.get_at_loaders_all_subjects(
                self.data_path)
        elif self.dataset == 'e4':
            loaders = dles.get_hr_and_acc_all_subjects(
                self.data_path, None, False)
        elif self.dataset == 'gr':
            ps = [1] * 2
            hertzes = [1.0/60] * 2
            n = 60 * 24 * 8
            loaders = {'e' + str(i): dlss.get_FPGL(n, ps, hertzes)
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

        print 'Initializing directories'

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

        self.wavelet_dir = rmu.init_dir(
            'wavelets',
            save,
            self.save_load_dir)

    def _load(self):

        print 'Loading wavelets'

        self.wavelets = rmu.get_wavelet_storage(
            self.num_views,
            self.num_sps,
            self.num_periods,
            self.subjects)

        for fn in os.listdir(self.wavelet_dir):
            path = os.path.join(self.wavelet_dir, fn)
            info = fn.split('_')
            s = info[1]
            p = int(info[3])
            sp = int(info[5])
            v = int(info[7])
            Yh_or_Yl = info[8]
            index = None
            coeffs = None

            with open(path) as f:
                loaded = np.load(f)

                if Yh_or_Yl == 'Yh':
                    index = 0
                    loaded = {int(h_fn.split('_')[1]) : a
                              for (h_fn, a) in loaded.items()}
                    num_coeffs = len(loaded)
                    coeffs = [loaded[i] 
                              for i in xrange(num_coeffs)]
                elif Yh_or_Yl == 'Yl':
                    index = 1
                    coeffs = loaded

            self.wavelets[s][p][sp][v][index] = coeffs

    def _compute(self):

        for subject in self.subjects:

            print 'Computing subperiod wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_sp_wavelet_transforms(subject)

            if self.save:
                self._save(Yls, Yhs, subject)

    def _save(self, Yls, Yhs, subject):

        for (v, (v_Yhs, v_Yls)) in enumerate(zip(Yhs, Yls)):
            for (p, (p_Yhs, p_Yls)) in enumerate(zip(v_Yhs, v_Yls)):
                for sp in xrange(self.num_sps):
                    path = '_'.join([
                        'subject',
                        subject,
                        'period',
                        str(p),
                        'subperiod',
                        str(sp),
                        'view',
                        str(v)])
                    l_fname = path + '_Yl_dtcwt_coefficients.thang'
                    l_path = os.path.join(
                        self.wavelet_dir, l_fname)

                    with open(l_path, 'w') as f:
                        np.save(f, p_Yls[sp])

                    h_fname = path + '_Yh_dtcwt_coefficients.thang'
                    h_path = os.path.join(
                        self.wavelet_dir, h_fname)

                    with open(h_path, 'w') as f:
                        np.savez(f, *p_Yhs[sp])

    def _get_sp_wavelet_transforms(self, subject):

        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        iterable = enumerate(zip(
            self.rates, self.servers[subject]))

        for (i, (r, ds)) in iterable:
            window = int(r * self.period)
            sp_window = int(self.subperiod * r)
            threshold = self.names[i] == 'TEMP'
            data = ds.get_data()

            for p in xrange(self.num_periods[subject]):
                data_p = data[p * window: (p+1) * window]

                Yls[i].append([])
                Yhs[i].append([])

                for sp in xrange(self.num_sps):
                    data_sp = data_p[sp * sp_window : (sp +1) * sp_window]
                    data_sp = get_non_nan(data_sp)[:,np.newaxis]

                    if threshold:
                        data_sp[data_sp > 40] = 40

                    (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                        data_sp, 
                        int(log(data_sp.shape[0], 2)) - 1,
                        self.biorthogonal,
                        self.qshift)

                    Yls[i][-1].append(Yl)
                    Yhs[i][-1].append(Yh)

        return (Yls, Yhs)
