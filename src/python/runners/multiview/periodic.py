import os

import numpy as np
import pandas as pd
import spancca as scca
import data.loaders.e4.shortcuts as dles

from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9, Oranges9
from bokeh.plotting import output_file, show
from bokeh.models.layouts import Column, Row
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering as AC
from multiprocessing import Pool
from math import log
from time import mktime
from datetime import datetime

class MVCCADTCWTRunner:

    def __init__(self, 
        biorthogonal,
        qshift,
        servers, 
        period,
        delay=None,
        correlation_dir=None,
        load_correlation=False,
        save_correlation=False,
        show_correlation=False,
        cca_dir=None,
        load_cca=False,
        save_cca=False,
        show_cca=False,
        kmeans=None,
        plot_dir=None):

        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.period = period
        self.delay = delay
        self.load_correlation = load_correlation
        self.save_correlation = save_correlation
        self.show_correlation = show_correlation

        if correlation_dir is None:
            self.correlation_dir = correlation_dir

            if self.load_correlation or self.save_correlation:
                raise ValueError(
                    'Directory path for correlation save/load must be provided.')
        else:
            dir_name = get_ts('_'.join(
                'delay',
                str(self.delay),
                'period',
                str(self.period)),
                'correlation')
            self.correlation_dir = os.path.join(
                correlation_dir,
                dir_name)

            os.mkdir(self.correlation_dir)

        self.load_cca = load_cca
        self.save_cca = save_cca
        self.show_cca = show_cca

        if cca_dir is None:
            self.cca_dir = cca_dir

            if self.load_cca or self.save_cca:
                raise ValueError(
                    'Directory path for cca save/load must be provided.')
        else:
            dir_name = get_ts('_'.join(
                'delay',
                str(self.delay),
                'period',
                str(self.period)),
                'cca')
            self.cca_dir = os.path.join(
                cca_dir,
                dir_name)

            os.mkdir(self.cca_dir)

        if plot_dir is None:
            self.plot_dir = plot_dir

            if self.show_cca or self.show_correlation:
                raise ValueError(
                    'Directory path for plots save/load must be provided.')

        else:
            dir_name = get_ts('_'.join(
                'delay',
                str(self.delay),
                'period',
                str(self.period)),
                'plots')
            self.plot_dir = os.path.join(
                plot_dir,
                dir_name)

            os.mkdir(self.plot_dir)

        self.servers = dles.get_hr_and_acc_all_subjects(
            self.hdf5_path)
        self.subjects = self.servers.keys()
        self.rates = [ds.get_status()['data_loader'].get_status()['hertz']
                      for ds in self.servers]
        self.num_periods = min(
            [ds.rows() / (r * self.period) 
             for (ds, r) in zip(self.servers, self.rates)])
        self.num_views = len(self.servers)

        get_spud = lambda d, nd: SPUD(
            self.num_views, 
            default=d, 
            no_double=nd)
        get_spud_list = lambda d, nd: [get_spud(d, nd)
                                   for i in xrange(self.num_periods)]

        self.wavelets = {s : [] for s in self.subjects}
        self.correlation= {s : get_spud_list(None, False) 
                           for s in self.subjects} 
        self.pw_cca_mag = {s : get_spud_list(dict, True)
                           for s in self.subjects}
        self.pw_cca_phase = {s : get_spud_list(dict, True)
                             for s in self.subjects}
        self.mv_cca_mag = {s : get_spud_list(dict, True)
                           for s in self.subjects}
        self.mv_cca_phase = {s : get_spud_list(dict, True)
                             for s in self.subjects}

    def run(self):

        if not self.load_correlation and not self.load_cca:
            for subject in self.subjects:
                (Yls, Yhs) = self._get_wavelet_transforms(subject)

                for period in xrange(self.num_periods):
                    Yhs_period = [view[period] for view in Yhs]
                    Yls_period = [view[period] for view in Yls]

                    self.wavelets[subject].append(
                        (Yhs_period, Yls_period))

        if self.load_correlation:
            self._load_correlation()
        else:
            self._compute_correlation()

        if self.load_cca:
            self._load_cca()
        else:
            self._compute_cca()

        if self.kmeans is not None:
            self._run_kmeans()

        if self.show_correlation:
            self._show_correlation()

        if self.show_cca:
            self._show_cca()

    def _load_cca(self):

        cca = {}

        for fn in os.listdir(self.cca_dir):
            path = os.path.join(self.cca_dir, fn)

            with open(path) as f:
                cca[fn] = np.load(f)

        for (k, mat) in cca.items():
            info = k.split('_')
            name = info[0]
            subject = info[2]
            period = int(info[6])
            views = [int(i) for i in info[8].split('-')]
            phase_or_mag = info[9]
            l = None

            if phase_or_mag == 'phase':
                l = self.pw_cca_phase
            elif phase_or_mag == 'mag':
                l = self.pw_cca_mag

            l[subject][period].get(
                views[0], views[1])[name] = mat
    
    def _load_correlation(self):

        correlation = {}

        for fn in os.listdir(self.correlation_dir):
            path = os.path.join(self.correlation_dir, fn)

            with open(path) as f:
                correlation[fn] = np.load(f)

        for (k, hm) in correlation.items():
            info = k.split('_')
            period = int(info[1])
            views = [int(i) for i in info[3].split('-')]
            
            self.correlation[period].insert(
                views[0], views[1], hm)

    def _run_kmeans(self):

        model = KMeans(n_clusters=self.kmeans, random_state=0).fit()

    def _compute_cca(self):

        for (period, (Yhs, Yls)) in enumerate(self.wavelets):
            current = SPUD(self.num_views, no_double=True)
            wavelet_matrices = [_get_sampled_wavelets(Yh, Yl)
                                for (Yh, Yl) in zip(Yhs, Yls)]
            wms_mag = [np.absolute(wm) 
                       for wm in wavelet_matrices]
            wms_phase = [np.angle(wm)
                         for wm in wavelet_matrices]
            current_mag = _get_pw_cca(wms_mag)
            current_phase = _get_pw_cca(wms_phase)

            self.pw_cca_mag.append(current_mag)
            self.pw_cca_phase.append(current_phase)

            if self.save_cca:
                self._save_cca(current_mag, period, 'mag')
                self._save_cca(current_phase, period, 'phase')

        # TODO: Do multi-view CCA on magnitude of coefficients

        # TODO: Do CCA on unmodified complex coefficients (probably not)

    def _save_cca(self, subject, current, period, phase_or_mag):

        for (k, xy_pair) in current.items():
            views_str = 'views_' + '-'.join([str(j) for j in k])
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
            for (period, (Yhs, Yls)) in enumerate(self.wavelets[subject]):
                correlation = self._get_period_correlation(
                    Yhs, Yls)

                self.correlation[subject].append(correlation)

                if self.save_correlation:
                    for (k, hm) in correlation.items():
                        views_str = 'views_' + '-'.join([str(i) for i in k])
                        path = '_'.join([
                            'subject',
                            subject,
                            'period',
                            str(period),
                            'views',
                            '-'.join([str(j) for j in k]),
                            phase_or_mag,
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

    def _get_wavelet_transforms(self):

        data = [ds.get_data() for ds in self.servers]

        factors = [int(self.period * r) for r in self.rates]

        if self.delay is not None:
            data = [view[self.delay * r:] 
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

    def _show_cca(self):

        for subject in self.subjects:
            timelines_mag = SPUD(
                self.num_views, default=list, no_double=True)
            timelines_phase = SPUD(
                self.num_views, default=list, no_double=True)

            for (i, period) in enumerate(self.pw_cca_mag[subject]):
                for (k, xy_pair) in period.items():
                    timelines_mag.get(k[0], k[1]).append(xy_pair)
                    
            for (i, period) in enumerate(self.pw_cca_phase[subject]):
                for (k, xy_pair) in period.items():
                    timelines_phase.get(k[0], k[1]).append(xy_pair)

            for (k, l) in timelines_mag.items():
                self._plot_cca(k, l, 'mag', subject)

            for (k, l) in timelines_phase.items():
                self._plot_cca(k, l, 'phase', subject)

    def _show_correlation(self):

        timelines = SPUD(self.num_views, default=list)
        prev = None

        for subject in self.subjects:
            for (i, period) in enumerate(self.correlation[subject]):
                for (k, hm) in period.items():
                    timelines.get(k[0], k[1]).append(hm)

                prev = period

            for (k, l) in timelines.items():
                self._plot_correlation(k, l, subject)

    def _plot_cca(self, key, timeline, phase_or_mag, subject):

        (i, j) = key
        (nx, px) = timeline[0]['Xw'].shape
        (ny, py) = timeline[0]['Yw'].shape
        names = [ds.name() for ds in self.servers]
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
            width=50*X_t.shape[1],
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
            width=50*X_t.shape[1],
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
        names = [ds.name() for ds in self.servers]
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
        filename = get_ts(
            'correlation_of_wavelet_coefficients_' +
            'subject',
            subject,
            phase_or_mag,
            names[i], 
            names[j]) + '.html'
        filepath = os.path.join(self.plot_dir, filename)

        output_file(
            filepath, 
            'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        show(plot)

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
