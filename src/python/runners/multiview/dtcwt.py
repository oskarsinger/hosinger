import os

import numpy as np
import spancca as scca
import data.loaders.e4.shortcuts as dles

from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.ml import get_kmeans
from data.servers.batch import BatchServer as BS
from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9, Oranges9
from bokeh.plotting import output_file, show
from bokeh.models.layouts import Column, Row
from multiprocessing import Pool
from math import log

class MVDTCWTRunner:

    def __init__(self, 
        hdf5_path,
        biorthogonal,
        qshift,
        period,
        subperiod=None,
        do_phase=False,
        delay=None,
        save_load_dir=None, 
        compute_wavelets=False,
        load_wavelets=False,
        save_wavelets=False,
        compute_sp_corr=False,
        load_sp_corr=False,
        save_sp_corr=False,
        show_sp_corr=False,
        vpw_corr_kmeans=None,
        vpw_cca_kmeans=None,
        show_kmeans=False):

        self.hdf5_path = hdf5_path
        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.period = period
        self.subperiod = subperiod
        self.compute_wavelets = compute_wavelets
        # TODO: k should probably be internally determined carefully
        self.show_kmeans = show_kmeans
        self.show_sp_correlation = show_sp_correlation 

        self._init_dirs(
            save_load_dir,
            load_wavelets,
            save_wavelets,
            load_sp_wavelets,
            save_sp_wavelets,
            load_sp_corr,
            save_sp_corr,
            show_sp_corr)
        self._init_server_stuff()

        self.wavelets = {s : [] for s in self.subjects}
        self.sp_wavelets = {s : [] for s in self.subjects}

        self.sp_correlation = self._get_list_spud_dict()
        self.mv_cca_mag = self._get_list_spud_dict(no_double=True)
        self.mv_cca_phase = self._get_list_spud_dict(no_double=True)

    def run(self):

        if self.compute_wavelets:
            self._compute_wavelet_transforms()

        if self.subperiod is not None:
            self._compute_sp_wavelet_transforms()
            self._compute_sp_correlation()

            if self.show_sp_correlation:
                self._show_sp_correlation()

    def _init_server_stuff(self):

        print 'Initializing servers'

        loaders = dles.get_hr_and_acc_all_subjects(
            self.hdf5_path, None, False)
        self.servers = {}

        for (s, dl_list) in loaders.items():
            # This is to ensure that all subjects have sufficient data
            s = s[-2:]
            try:
                [dl.get_data() for dl in dl_list]

                self.servers[s] = [BS(dl) for dl in dl_list]
            except Exception, e:
                print 'Could not load data for subject', s
                print e
        
        (self.rates, self.names) = unzip(
            [(dl.get_status()['hertz'], dl.name())
             for dl in loaders.items()[0][1]])
                    
        self.subjects = self.servers.keys()
        server_np = lambda ds, r: ds.rows() / (r * self.period)
        subject_np = lambda s: min(
            [server_np(ds, r) 
             for (ds, r) in zip(self.servers[s], self.rates)])
        self.num_periods = {s : int(subject_np(s))
                            for s in self.subjects}
        self.num_views = len(self.servers.items()[0][1])

    def _init_dirs(self,
        save_load_dir,
        load_wavelets,
        save_wavelets,
        load_sp_wavelets,
        save_sp_wavelets,
        load_sp_corr,
        save_sp_corr,
        show_sp_corr)

        print 'Initializing directories'

        self.save_load_dir = save_load_dir 
        self.load_wavelets = load_wavelets 
        self.save_wavelets = save_wavelets 
        self.load_sp_wavelets = load_sp_wavelets 
        self.save_sp_wavelets = save_sp_wavelets 
        self.load_sp_corr = load_sp_corr 
        self.save_sp_corr = save_sp_corr 
        self.show_sp_corr = show_sp_corr 

        show = show_sp_corr

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

        self.sp_corr_dir = self._init_dir(
            'sp_corr',
            save_correlation)
        self.wavelet_dir = self._init_dir(
            'wavelets',
            save_wavelets)

    def _load_wavelets(self):

        print 'Loading wavelets'

        wavelets = {}

        for fn in os.listdir(self.wavelet_dir):
            path = os.path.join(self.wavelet_dir, fn)

            with open(path) as f:
                wavelets[fn] = np.load(f)

        get_p = lambda: [[None, None] for i in xrange(self.num_views)]
        get_s = lambda s: [get_period() 
                           for i in xrange(self.num_periods[s])]
        self.wavelets = {s : get_s(s)
                         for s in self.subjects}

        for (k, loaded) in wavelets.items():
            info = k.split('_')
            s = info[1]
            p = int(info[3])
            v = int(info[5])
            Yh_or_Yl = info[6]
            coeffs = None
            index = None

            if Yh_or_Yl == 'Yh':
                coeffs = unzip(loaded.items())[1]
                index = 0
            elif Yh_or_Yl == 'Yl':
                coeffs = loaded
                index = 1

            self.wavelets[s][p][v][index] = coeffs

    # TODO: Do multi-view CCA on magnitude and phase of coefficients
    # Probably use CCALin
    def _compute_sp_correlation(self):

        for subject in self.subjects:

            sp_wavelets = self.sp_wavelets[subject]

            for (p, (Yhs, Yss)) in enumerate(sp_wavelets):
                print 'Stuff'

    def _compute_wavelet_transforms(self):

        for subject in self.subjects:

            print 'Computing wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_wavelet_transforms(subject)

            for period in xrange(self.num_periods[subject]):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.wavelets[subject].append(
                    (Yhs_period, Yls_period))

                if self.save_wavelets:
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

        if self.delay is not None:
            data = [view[int(self.delay * r):] 
                    for (r,view) in zip(self.rates, data)]

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

    def _compute_sp_wavelet_transforms(self):

        for subject in self.subjects:

            print 'Computing subperiod wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_sp_wavelet_transforms(subject)

            for period in xrange(self.num_periods[subject]):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.sp_wavelets[subject].append(
                    (Yhs_period, Yls_period))

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
                             for (view, sp_f) in zip(data, sp_factors)]
            j = 0
            
            # Add a list to each view for the current super period
            for i in xrange(self.num_views):
                Yls[i].append([])
                Yhs[i].append([])

            while not sp_complete: 
                sp_exceeded = [(j+1) >= sp_t for sp_t in sp_thresholds]
                sp_complete = any(sp_exceeded)
                sp_current_data = [view[j*sp_f:(j+1)*sp_f]
                                   for (sp_f, view) in zip(sp_factors, current_data)]

                for (i, view) in enumerate(sp_current_data):
                    (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                        view, 
                        int(log(view.shape[0], 2)) - 2,
                        self.biorthogonal,
                        self.qshift)
                    
                    # Append the current subperiod's wavelets to the current period
                    Yls[i][-1].append(Yl)
                    Yhs[i][-1].append(Yh)
               
                j += 1

            k += 1

        return (Yls, Yhs)

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
