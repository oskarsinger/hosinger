import os
import json

import numpy as np
import data.loaders.e4.shortcuts as dles

from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from data.servers.batch import BatchServer as BS
from wavelets import dtcwt
from multiprocessing import Pool
from math import log

class MVDTCWTSPRunner:

    def __init__(self, 
        hdf5_path,
        save_load_dir=None, 
        save=False,
        load=False):

        self.hdf5_path = hdf5_path
        self.period = 24 * 3600
        self.subperiod = 3600
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

        loaders = dles.get_hr_and_acc_all_subjects(
            self.hdf5_path, None, False)
        self.servers = {}

        for (s, dl_list) in loaders.items():
            s = s[-2:]

            # This is to ensure that all subjects have sufficient data
            # Only necessary if we have to recompute wavelets
            try:
                if not self.load:
                    [dl.get_data() for dl in dl_list]

                self.servers[s] = [BS(dl) for dl in dl_list]
            except Exception, e:
                print 'Could not load data for subject', s
                print e
        
        (self.rates, self.names) = unzip(
            [(dl.get_status()['hertz'], dl.name())
             for dl in loaders.items()[0][1]])
                    
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
                f.write(np.json)
        else:
            path = os.path.join(
                self.save_load_dir,
                'num_periods.json')

            with open(path) as f:
                l = f.readline().strip()
                self.num_periods = json.loads(l)

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
                'MVDTCWTSPR',
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

        self.wavelet_dir = self._init_dir(
            'wavelets',
            save,
            self.save_load_dir)

    def _load(self):

        print 'Loading wavelets'

        get_sp = lambda: [[None, None] 
                          for i in xrange(self.num_views)]
        get_p = lambda: [get_sp() 
                         for i in xrange(self.num_sps)]
        get_s = lambda s: [get_p() 
                           for i in xrange(self.num_periods[s])]
        self.wavelets = {s : get_s(s)
                         for s in self.subjects}

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
                    loaded = {int(fn.split('_')[1]) : a
                              for (fn, a) in loaded.items()}
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
                for (v, (v_Yhs, v_Yls)) in enumerate(zip(Yhs, Yls)):
                    for (p, (p_Yhs, p_Yls)) in enumerate(zip(v_Yhs, v_Yhs)):
                        for (sp, Yl) in enumerate(p_Yls):
                            path = '_'.join([
                                'subject',
                                subject,
                                'period',
                                str(p),
                                'subperiod',
                                str(sp),
                                'view',
                                str(v),
                                'Yl_dtcwt_coefficients.thang'])

                            path = os.path.join(self.wavelet_dir, path)

                            with open(path, 'w') as f:
                                np.save(f, Yl)

                        for (sp, Yh) in enumerate(p_Yhs):
                            path = '_'.join([
                                'subject',
                                subject,
                                'period',
                                str(p),
                                'subperiod',
                                str(sp),
                                'view',
                                str(v),
                                'Yh_dtcwt_coefficients.thang'])

                            path = os.path.join(self.wavelet_dir, path)

                            with open(path, 'w') as f:
                                np.savez(f, *Yh)


    def _get_sp_wavelet_transforms(self, subject):

        data = [ds.get_data() for ds in self.servers[subject]]
        factors = [int(self.period * r) for r in self.rates]
        sp_factors = [int(self.subperiod * r) for r in self.rates]

        if self.delay is not None:
            data = [view[int(self.delay * r):] 
                    for (r,view) in zip(self.rates, data)]

        thresholds = [int(view.shape[0] * 1.0 / f)
                      for (view, f) in zip(data, factors)]
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        complete = False
        k = 0

        while not complete:
            exceeded = [(k+1) >= t for t in thresholds]
            complete = any(exceeded)
            current_data = [view[k*f:(k+1)*f]
                            for (f, view) in zip(factors, data)]
            sp_thresholds = [int(view.shape[0] * 1.0 / sp_f)
                             for (view, f) in zip(data, sp_factors)]
            j = 0
            
            # Add a list to each view for the current super period
            for i in xrange(self.num_views):
                Yls[i].append([])
                Yhs[i].append([])

            while not sp_complete: 
                exceeded = [(j+1) >= t for t in sp_thresholds]
                sp_complete = any(exceeded)
                iterable = zip(sp_factors, current_data)
                sp_current_data = [view[j*f:(j+1)*f]
                                   for (f, view) in iterable]

                for (i, view) in enumerate(current_data):
                    (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                        view, 
                        int(log(view.shape[0], 2)) - 2,
                        self.biorthogonal,
                        self.qshift)
                    
                    Yls[i][-1].append(Yl)
                    Yhs[i][-1].append(Yh)
               
                j += 1

            k += 1

        return (Yls, Yhs)
