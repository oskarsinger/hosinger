import os

import numpy as np
import pandas as pd
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
from sklearn.cross_decomposition import CCA
from multiprocessing import Pool
from math import log
from time import mktime
from datetime import datetime

class MVCCADTCWTRunner:

    def __init__(self, 
        hdf5_path,
        biorthogonal,
        qshift,
        period,
        sub_period,
        delay=None,
        save_load_dir=None, 
        compute_correlation=False,
        load_correlation=False,
        save_correlation=False,
        show_correlation=False,
        compute_cca=False,
        load_cca=False,
        save_cca=False,
        show_cca=False,
        correlation_kmeans=None,
        cca_kmeans=None):

        self.hdf5_path = hdf5_path
        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.period = period
        self.delay = delay
        self.compute_correlation = compute_correlation
        self.compute_cca = compute_cca
        # TODO: k should probably be internally determined carefully
        self.correlation_kmeans = correlation_kmeans
        self.cca_kmeans = cca_kmeans

        self._init_dirs(
            save_load_dir,
            load_correlation,
            save_correlation,
            show_correlation,
            load_cca,
            save_cca,
            show_cca)
        self._init_server_stuff()

        self.wavelets = {s : [] for s in self.subjects}
        self.sp_wavelets = {s : [] for s in self.subjects}

        self.correlation = self._get_list_spud_dict()
        self.sp_correlation = self._get_list_spud_dict()

        self.pw_cca_mag = self._get_list_spud_dict(no_double=True)
        self.pw_cca_phase = self._get_list_spud_dict(no_double=True)
        self.mv_cca_mag = self._get_list_spud_dict(no_double=True)
        self.mv_cca_phase = self._get_list_spud_dict(no_double=True)

        self.corr_kmeans_mag = self._get_list_spud_dict()
        self.corr_kmeans_phase = self._get_list_spud_dict()
        self.pw_cca_mag_kmeans = self._get_list_spud_dict(no_double=True)
        self.pw_cca_phase_kmeans = self._get_list_spud_dict(no_double=True)

    def run(self):

        if not self.load_correlation and not self.load_cca:
            self._compute_wavelet_transforms()

        if self.load_correlation:
            self._load_correlation()
        elif self.compute_correlation:
            self._compute_correlation()

        if self.load_cca:
            self._load_cca()
        elif self.compute_cca:
            self._compute_cca()

        if self.correlation_kmeans is not None:
            self._compute_correlation_kmeans()
            self._show_kmeans(
                'Correlation Magnitude',
                self.corr_kmeans_mag)
            self._show_kmeans(
                'Correlation Phase',
                self.corr_kmeans_phase)

        if self.cca_kmeans is not None:
            self._compute_cca_kmeans()
            self._show_kmeans(
                'CCA Magnitude',
                self.pw_cca_mag_kmeans)
            self._show_kmeans(
                'CCA Phase',
                self.pw_cca_phase_kmeans)

        if self.show_correlation:
            self._show_correlation()

        if self.show_cca:
            self._show_cca()

    def _init_server_stuff(self):

        print 'Initializing servers'

        loaders = dles.get_hr_and_acc_all_subjects(
            self.hdf5_path, None, False)
        self.servers = {}

        for (s, dl_list) in loaders.items()[-7:]:
            # This is to ensure that all subjects have sufficient data
            s = s[-2:]
            try:
                [dl.get_data() for dl in dl_list]

                self.servers[s] = [BS(dl) for dl in dl_list]
            except Exception, e:
                print 'Could not load data for subject', s
                print e
        
        self.rates = [dl.get_status()['hertz']
                      for dl in loaders.items()[0][1]]
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
        load_correlation,
        save_correlation,
        show_correlation,
        load_cca,
        save_cca,
        show_cca):

        print 'Initializing directories'

        self.load_correlation = load_correlation
        self.save_correlation = save_correlation
        self.show_correlation = show_correlation
        self.load_cca = load_cca
        self.save_cca = save_cca
        self.show_cca = show_cca

        save = save_correlation or save_cca
        load = load_correlation or load_cca

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('_'.join([
                'MVCCADTCWT',
                'delay',
                str(self.delay),
                'period',
                str(self.period)]))

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.correlation_dir = self._init_dir(
            'correlation',
            save_correlation)
        self.cca_dir = self._init_dir(
            'cca',
            save_cca)
        self.plot_dir = self._init_dir(
            'plots',
            show_cca or show_correlation)

    def _init_dir(self, dir_name, save):

        dir_path = os.path.join(
            self.save_load_dir,
            dir_name)

        if save and not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        return dir_path

    def _get_list_spud_dict(self, no_double=False):

        get_list_spud = lambda nd: SPUD(
            self.num_views, default=list, no_double=nd)

        return {s : get_list_spud(no_double)
                for s in self.subjects}

    def _load_cca(self):

        print 'Loading CCA'

        cca = {}

        for fn in os.listdir(self.cca_dir):
            path = os.path.join(self.cca_dir, fn)

            with open(path) as f:
                cca[fn] = np.load(f)

        keys = self.pw_cca_mag[self.subjects[0]].keys()

        for subject in self.subjects:
            num_periods = self.num_periods[subject]

            for (i, j) in keys:
                self.pw_cca_mag[subject].insert(
                    i, j, [{} for k in xrange(num_periods)])
                self.pw_cca_phase[subject].insert(
                    i, j, [{} for k in xrange(num_periods)])

        for (k, mat) in cca.items():
            info = k.split('_')
            name = info[0]
            subject = info[2]
            period = int(info[4])
            views = [int(i) for i in info[6].split('-')]
            phase_or_mag = info[7]
            l = None

            if phase_or_mag == 'phase':
                l = self.pw_cca_phase
            elif phase_or_mag == 'mag':
                l = self.pw_cca_mag

            l[subject].get(
                views[0], views[1])[period][name] = mat
    
    def _load_correlation(self):

        print 'Loading correlation'

        correlation = {}

        for fn in os.listdir(self.correlation_dir):
            path = os.path.join(self.correlation_dir, fn)

            with open(path) as f:
                correlation[fn] = np.load(f)

        for (k, hm) in correlation.items():
            info = k.split('_')
            subject = info[1]
            period = int(info[3])
            views = [int(i) for i in info[5].split('-')]
            
            self.correlation[subject].get(
                views[0], views[1]).append(hm)

    def _compute_cca_kmeans(self):

        print 'Computing CCA k-means'

        mag_data = SPUD(
            self.num_views, default=list, no_double=True)
        phase_data = SPUD(
            self.num_views, default=list, no_double=True)
        subjects = SPUD(
            self.num_views, default=list, no_double=True)

        for subject in self.subjects:
            mag = self.pw_cca_mag[subject]
            phase = self.pw_cca_mag[subject]

            for ((i, j), mag_pairs) in mag.items():
                phase_pairs = phase.get(i, j)
                # TODO: Double check that the dimension manipulation is correct here
                stack = lambda p: np.ravel(
                    np.vstack([p['Xw'], p['Yw']]))
                mag_l = [stack(p) for p in mag_pairs]
                phase_l = [stack(p) for p in phase_pairs]

                mag_data.get(i, j).extend(mag_l)
                phase_data.get(i, j).extend(phase_l)
                subjects.get(i, j).extend(
                    [subject] * len(mag_l))

        mag = SPUD(self.num_views, no_double=True)
        phase = SPUD(self.num_views, no_double=True)

        for ((i, j), mag_v) in mag_data.items():
            mag_as_rows = np.vstack(mag_v)
            phase_as_rows = np.vstack(
                phase_data.get(i, j))

            mag.insert(i, j, mag_as_rows)
            phase.insert(i, j, phase_as_rows)

        self.pw_cca_mag_kmeans = self._get_kmeans_spud_dict(
            mag, subjects, self.cca_kmeans)
        self.pw_cca_phase_kmeans = self._get_kmeans_spud_dict(
            phase, subjects, self.cca_kmeans)

    def _compute_correlation_kmeans(self):

        print 'Computing correlation k-means'

        data = SPUD(self.num_views, default=list)
        subjects = SPUD(self.num_views, default=list)
        
        for subject in self.subjects:
            items = self.correlation[subject].items()
            
            for ((i, j), corr_list) in items:
                raveled_corrs = [np.ravel(corr) for corr in corr_list]

                data.get(i, j).extend(
                    [np.ravel(corr) for corr in corr_list])
                subjects.get(i, j).extend(
                    [subject] * len(corr_list))

        mag = SPUD(self.num_views)
        phase = SPUD(self.num_views)

        for ((i, j), v) in data.items():
            corrs_as_rows = np.vstack(v)

            mag.insert(i, j, np.absolute(corrs_as_rows))
            phase.insert(i, j, np.angle(corrs_as_rows))

        self.corr_kmeans_mag = self._get_kmeans_spud_dict(
            mag, subjects, self.correlation_kmeans)
        self.corr_kmeans_phase = self._get_kmeans_spud_dict(
            phase, subjects, self.correlation_kmeans)

    def _get_kmeans_spud_dict(self, data, subjects, k, no_double=False):

        label_spud = self._get_list_spud_dict(no_double=no_double)

        for ((i, j), d) in data.items():
            labels = get_kmeans(
                d, k=k).labels_.tolist()
            subject_list = subjects.get(i, j)

            for (l, s) in zip(labels, subject_list):
                label_spud[s].get(i, j).append(l)

        return label_spud

    def _compute_cca(self):

        for subject in self.subjects:

            print 'Computing CCA for subject', subject

            for (period, (Yhs, Yls)) in enumerate(self.wavelets[subject]):
                current = SPUD(self.num_views, no_double=True)
                wavelet_matrices = [_get_sampled_wavelets(Yh, Yl)
                                    for (Yh, Yl) in zip(Yhs, Yls)]
                wms_mag = [np.absolute(wm) 
                           for wm in wavelet_matrices]
                wms_phase = [np.angle(wm)
                             for wm in wavelet_matrices]
                current_mag = _get_pw_cca(wms_mag)
                current_phase = _get_pw_cca(wms_phase)

                self.pw_cca_mag[subject] = _get_appended_spud(
                    self.pw_cca_mag[subject], current_mag)
                self.pw_cca_phase[subject] = _get_appended_spud(
                    self.pw_cca_phase[subject], current_phase)

                if self.save_cca:
                    self._save_cca(subject, current_mag, period, 'mag')
                    self._save_cca(subject, current_phase, period, 'phase')

        # TODO: Do multi-view CCA on magnitude and phase of coefficients
        # Probably use CCALin

    def _save_cca(self, subject, current, period, phase_or_mag):

        for (k, xy_pair) in current.items():
            path = '_'.join([
                'subject',
                subject,
                'period',
                str(period),
                'views',
                '-'.join([str(j) for j in k]),
                phase_or_mag,
                'dtcwt_cca_matrix.thang'])

            for (l, mat) in xy_pair.items():
                if self.cca_dir is not None:
                    current_path = os.path.join(
                        self.cca_dir, l + '_' + path)

                with open(current_path, 'w') as f:
                    np.save(f, mat)

    def _compute_correlation(self):

        for subject in self.subjects:

            print 'Computing correlation for subject', subject

            for (period, (Yhs, Yls)) in enumerate(self.wavelets[subject]):
                correlation = self._get_period_correlation(
                    Yhs, Yls)

                for ((i, j), corr) in correlation.items():
                    self.correlation[subject].get(i, j).append(corr)

                if self.save_correlation:
                    for (k, hm) in correlation.items():
                        path = '_'.join([
                            'subject',
                            subject,
                            'period',
                            str(period),
                            'views',
                            '-'.join([str(j) for j in k]),
                            'dtcwt_correlation_matrix.thang'])

                        path = os.path.join(self.correlation_dir, path)

                        with open(path, 'w') as f:
                            np.save(f, hm)

    def _get_period_correlation(self, Yhs, Yls):

        Yh_matrices = [_get_sampled_wavelets(Yh, Yl)
                       for (Yh, Yl) in zip(Yhs, Yls)]
        min_length = min(
            [Y.shape[0] for Y in Yh_matrices]) 
        rates = [int(Y.shape[0] / min_length)
                 for Y in Yh_matrices]
        subsamples = [m[::r,:]
                      for (m, r) in zip(Yh_matrices, rates)]
        get_matrix = lambda i,j: np.dot(
            subsamples[i].T, subsamples[j])
        correlation = SPUD(self.num_views)

        for (i, j) in correlation.keys():
            mat = get_matrix(i, j)
            correlation.insert(i, j, mat)

        return correlation

    def _compute_wavelet_transforms(self):

        for subject in self.subjects:

            print 'Computing wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_wavelet_transforms(subject)

            for period in xrange(self.num_periods[subject]):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.wavelets[subject].append(
                    (Yhs_period, Yls_period))

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

    def _get_sp_wavelet_transforms(self):

        print 'Stuff' 

    def _get_resampled_data(self):

        p = Pool(len(self.servers))
        processes = []
        resampled = []

        for (ds, rate) in zip(self.servers, self.rates):
            processes.append(p.apply_async(
                _get_resampled_view, (ds, rate)))

        for process in processes:
            resampled.append(process.get())

        return resampled

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

    def _show_cca(self):

        for subject in self.subjects:

            print 'Producing CCA plots for subject', subject

            for (k, l) in self.pw_cca_mag[subject].items():
                self._plot_cca(k, l, 'mag', subject)

            for (k, l) in self.pw_cca_phase[subject].items():
                self._plot_cca(k, l, 'phase', subject)

    def _show_correlation(self):

        timelines = SPUD(self.num_views, default=list)

        for subject in self.subjects:

            print 'Producing correlation plots for subject', subject

            for (k, l) in self.correlation[subject].items():
                self._plot_correlation(k, l, subject)

    def _plot_cca(self, key, timeline, phase_or_mag, subject):

        (i, j) = key
        print timeline[0].keys()
        (nx, px) = timeline[0]['Xw'].shape
        (ny, py) = timeline[0]['Yw'].shape
        names = [ds.name() for ds in self.servers[self.subjects[0]]]
        title = 'CCA decomposition of ' + phase_or_mag + \
            ' of views ' + \
            names[i] + ' and ' + names[j] + \
            ' by decimation level' + \
            ' for subject ' + subject
        x_title = 'CCA transform for view' + names[i]
        y_title = 'CCA transform for view' + names[j]
        x_name = 'period'
        y_name = 'decimation level'
        x_labels = [str(k) for k in xrange(len(timeline))]
        yx_labels = ['2^' + str(-k) for k in xrange(nx)]
        yy_labels = ['2^' + str(-k) for k in xrange(ny)]
        val_name = 'cca parameter'
        plots = []

        X_t = np.hstack(
            [t['Xw'] for t in timeline])
        Y_t = np.hstack(
            [t['Yw'] for t in timeline])

        X_pos_color_scheme = list(reversed(BuPu9))
        X_neg_color_scheme = list(reversed(Oranges9))
        X_plot = plot_matrix_heat(
            X_t,
            x_labels,
            yx_labels,
            title,
            x_name,
            y_name,
            val_name,
            width=150*X_t.shape[1],
            height=50*X_t.shape[0],
            pos_color_scheme=X_pos_color_scheme,
            neg_color_scheme=X_neg_color_scheme,
            norm_axis=0)
        Y_plot = plot_matrix_heat(
            Y_t,
            x_labels,
            yy_labels,
            title,
            x_name,
            y_name,
            val_name,
            width=150*X_t.shape[1],
            height=50*X_t.shape[0],
            norm_axis=0)

        plot = Column(*[X_plot, Y_plot])
        filename = get_ts('_'.join([
            'cca_of_wavelet_coefficients',
            'subject',
            subject,
            phase_or_mag,
            names[i], 
            names[j]])) + '.html'
        filepath = os.path.join(self.plot_dir, filename)

        output_file(
            filepath, 
            'cca_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        show(plot)

    def _plot_correlation(self, key, timeline, subject):

        (i, j) = key
        (n, p) = timeline[0].shape
        names = [ds.name() for ds in self.servers[self.subjects[0]]]
        title = 'Correlation of views ' + \
            names[i] + ' and ' + names[j] + \
            ' by decimation level' + \
            ' for subject ' + subject
        x_name = 'decimation level'
        y_name = 'decimation level'
        x_labels = ['2^' + str(-k) for k in xrange(p)]
        y_labels = ['2^' + str(-k) for k in xrange(n)]
        val_name = 'correlation'
        plots = []

        for (l, hm) in enumerate(timeline):
            pos_color_scheme = None
            neg_color_scheme = None
            hmp = plot_matrix_heat(
                hm,
                x_labels,
                y_labels,
                title,
                x_name,
                y_name,
                val_name,
                pos_color_scheme=pos_color_scheme,
                neg_color_scheme=neg_color_scheme,
                width=p*50,
                height=n*50)

            plots.append(hmp)

        plot = Column(*plots)
        filename = get_ts('_'.join([
            'correlation_of_wavelet_coefficients',
            'subject',
            subject,
            names[i], 
            names[j]])) + '.html'
        filepath = os.path.join(self.plot_dir, filename)

        output_file(
            filepath, 
            'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        show(plot)

def _get_appended_spud(list_spud, item_spud):

    for ((i, j), v) in item_spud.items():
        list_spud.get(i, j).append(v)

    return list_spud

def _get_pw_cca(views):

    num_views = len(views)
    current = SPUD(num_views, no_double=True)

    for i in xrange(num_views):
        for j in xrange(i+1, num_views):
            X_data = views[i]
            Y_data = views[j]
            cca = CCA(n_components=1)

            cca.fit(X_data, Y_data)

            xy_pair = {
                'Xw': cca.x_weights_,
                'Yw': cca.y_weights_}

            current.insert(i, j, xy_pair)

    return current

def _get_sampled_wavelets(Yh, Yl):

    # TODO: figure out what to do with Yl
    hi_and_lo = Yh# + [Yl]

    # Truncate for full-rank down-sampled coefficient matrix
    threshold = log(hi_and_lo[0].shape[0], 2)
    k = 1

    while log(k, 2) + k <= threshold:
        k += 1

    hi_and_lo = hi_and_lo[:k]
    basis = np.zeros(
        (hi_and_lo[-1].shape[0], k),
        dtype=complex)
    
    for (i, y) in enumerate(hi_and_lo):
        power = k - i - 1
        basis[:,i] = np.copy(y[::2**power,0])

    return basis

def _get_resampled_view(server, rate):

    data = server.get_data()
    loader = server.get_status()['data_loader']
    dt = loader.get_status()['start_times'][0]
    freq = 1.0 / rate
    dt_index = pd.DatetimeIndex(
        data=_get_dt_index(server.rows(), freq, dt))
    series = pd.Series(data=data[:,0], index=dt_index)

    return series.resample('S').pad().as_matrix()

def _get_dt_index(num_rows, factor, datetime):

    start = mktime(datetime.timetuple())
    times = (np.arange(num_rows) * factor + start).tolist()

    return [datetime.fromtimestamp(t)
            for t in times]
