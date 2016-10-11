import os
import json

import numpy as np
import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt
import utils as rmu

from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from data.servers.batch import BatchServer as BS
from wavelets import dtcwt
from multiprocessing import Pool
from math import log

class MVDTCWTRunner:

    def __init__(self, 
        hdf5_path,
        save_load_dir=None, 
        save=False,
        load=False):

        self.hdf5_path = hdf5_path
        self.period = 24 * 3600

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
                f.write(np_json)
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

        self.save_load_dir = save_load_dir 
        self.load= load
        self.save= save

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('_'.join([
                'MVDTCWTR',
                'period',
                str(self.period)]))

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

        get_p = lambda: [[None, None] 
                         for i in xrange(self.num_views)]
        get_s = lambda s: [get_p() 
                           for i in xrange(self.num_periods[s])]
        self.wavelets = {s : get_s(s)
                         for s in self.subjects}

        for fn in os.listdir(self.wavelet_dir):
            path = os.path.join(self.wavelet_dir, fn)
            info = fn.split('_')
            s = info[1]
            p = int(info[3])
            v = int(info[5])
            Yh_or_Yl = info[6]
            index = None
            coeffs = None

            with open(path) as f:
                loaded = np.load(f)

                if Yh_or_Yl == 'Yh':
                    index = 0
                    print unzip(loaded.items())[0]
                    coeffs = unzip(loaded.items())[1]
                elif Yh_or_Yl == 'Yl':
                    index = 1
                    coeffs = loaded

            self.wavelets[s][p][v][index] = coeffs

    def _compute(self):

        for subject in self.subjects:

            print 'Computing wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_wavelet_transforms(subject)

            for period in xrange(self.num_periods[subject]):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.wavelets[subject].append(
                    (Yhs_period, Yls_period))

                if self.save:
                    for (view, Yl) in enumerate(Yls_period):
                        path = '_'.join([
                            'subject',
                            subject,
                            'period',
                            str(period),
                            'view',
                            str(view),
                            'Yl_dtcwt_coefficients.thang'])

                        path = os.path.join(self.wavelet_dir, path)

                        with open(path, 'w') as f:
                            np.save(f, Yl)

                    for (view, Yh) in enumerate(Yhs_period):
                        path = '_'.join([
                            'subject',
                            subject,
                            'period',
                            str(period),
                            'view',
                            str(view),
                            'Yh_dtcwt_coefficients.thang'])

                        path = os.path.join(self.wavelet_dir, path)

                        with open(path, 'w') as f:
                            np.savez(f, *Yh)

    def _get_wavelet_transforms(self, subject):

        data = [ds.get_data() for ds in self.servers[subject]]
        factors = [int(self.period * r) for r in self.rates]
        thresholds  = [int(view.shape[0] * 1.0 / f)
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
            p = Pool(len(current_data))
            processes = []

            for (i, view) in enumerate(current_data):
                biorthogonal = {k : np.copy(v) 
                                for (k,v) in self.biorthogonal.items()}
                qshift = {k : np.copy(v) 
                          for (k,v) in self.qshift.items()}

                processes.append(p.apply_async(
                    dtcwt.oned.dtwavexfm,
                    (view, 
                    int(log(view.shape[0], 2)) - 2,
                    biorthogonal, 
                    qshift)))

            for (i, process) in enumerate(processes):
                (Yl, Yh, _) = process.get()

                Yls[i].append(Yl)
                Yhs[i].append(Yh)

            k += 1

        return (Yls, Yhs)

    """
    def _show_kmeans(self, title, labels):

        print title, 'KMeans Labels'

        max_periods = max(self.num_periods.values())
        default = lambda: [{} for i in xrange(max_periods)]
        by_period = SPUD(self.num_views, default=default)

        print '\tBy Subject'
        for subject in self.subjects:
            print '\t\tSubject:', subject
            spud = labels[subject]

            for ((i, j), timeline) in spud.items():
                print '\t\t\tView Pair:', i, j

                line = '\t'.join(
                    [str(p) + ': ' + str(label)
                     for (p, label) in enumerate(timeline)])

                print '\t\t\t\t' + line

                for p in xrange(max_periods):
                    if len(timeline) - 1 < p:
                        label = 'X'
                    else:
                        label = timeline[p]

                    by_period.get(i, j)[p][subject] = label

        print '\tBy Period'
        for ((i, j), timeline) in by_period.items():
            print '\t\tViews', i, j

            for (p, labels) in enumerate(timeline):
                print '\t\t\tPeriod', p

                line = '\t'.join(
                    [subject + ': ' + str(label)
                     for (subject, label) in labels.items()])

                print '\t\t\t\t' + line
    """
