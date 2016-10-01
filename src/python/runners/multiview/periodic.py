import os

import numpy as np
import pandas as pd
import spancca as scca

from multiprocessing import Pool
from drrobert.misc import unzip
from drrobert.file_io import get_timestamped as get_ts
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9, Oranges9
from bokeh.plotting import output_file, show
from bokeh.models.layouts import Column, Row
from sklearn.cross_decomposition import CCA
from math import log
from time import mktime
from datetime import datetime

class MVCCADTCWTRunner:

    def __init__(self, 
        biorthogonal,
        qshift,
        servers, 
        period,
        correlation_dir=None,
        load_correlation=False,
        save_correlation=False,
        show_correlation=False,
        cca_dir=None,
        load_cca=False,
        save_cca=False,
        show_cca=False,
        plot_path='.'):

        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.servers = servers
        self.period = period
        self.correlation_dir = correlation_dir
        self.load_correlation = load_correlation
        self.save_correlation = save_correlation
        self.show_correlation = show_correlation
        self.cca_dir = cca_dir
        self.load_cca = load_cca
        self.save_cca = save_cca
        self.show_cca = show_cca
        self.plot_path = plot_path

        self.rates = [ds.get_status()['data_loader'].get_status()['hertz']
                      for ds in self.servers]
        self.num_views = len(self.servers)
        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0
        self.wavelets = []
        self.correlation= [] 
        self.pairwise_cca = []

    def run(self):

        if not self.load_correlation and not self.load_cca:
            (Yls, Yhs) = self._get_wavelet_transforms()

            for period in xrange(len(Yhs[0])):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.wavelets.append(
                    (Yhs_period, Yls_period))

        if self.load_correlation:
            self._load_correlation()
        else:
            self._compute_correlation()

        if self.load_cca:
            self._load_cca()
        else:
            self._compute_cca()

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

        num_periods = max(
            [int(k.split('_')[2])
             for k in cca.keys()])

        self.pairwise_cca = [SPUD(
                                self.num_views, 
                                default=dict, 
                                no_double=True)
                             for i in xrange(num_periods)]

        for (k, mat) in cca.items():
            info = k.split('_')
            name = info[0]
            period = int(info[2]) - 1
            views = [int(i) for i in info[4].split('-')]

            self.pairwise_cca[period].get(
                views[0], views[1])[name] = mat
    
    def _load_correlation(self):

        correlation = {}

        for fn in os.listdir(self.correlation_dir):
            path = os.path.join(self.correlation_dir, fn)

            with open(path) as f:
                correlation[fn] = np.load(f)

        num_periods = max(
            [int(k.split('_')[1]) 
             for k in correlation.keys()])

        self.correlation = [SPUD(self.num_views)
                            for i in xrange(num_periods)]

        for (k, hm) in correlation.items():
            info = k.split('_')
            period = int(info[1]) - 1
            views = [int(i) for i in info[3].split('-')]

            self.correlation[period].insert(
                views[0], views[1], hm)

    def _compute_cca(self):

        for (period, (Yhs, Yls)) in enumerate(self.wavelets):
            current = SPUD(self.num_views, no_double=True)
            wavelet_matrices = [_get_sampled_wavelets(Yh, Yl)
                                for (Yh, Yl) in zip(Yhs, Yls)]
            abs_wms = [np.absolute(wm) 
                       for wm in wavelet_matrices]

            for i in xrange(self.num_views):
                for j in xrange(i+1, self.num_views):
                    X_data = abs_wms[i]
                    Y_data = abs_wms[j]
                    cca = CCA(n_components=1)

                    cca.fit(X_data, Y_data)

                    xy_pair = {
                        'Xw': cca.x_weights_,
                        'Yw': cca.y_weights_}

                    current.insert(i, j, xy_pair)

            self.pairwise_cca.append(current)

            if self.save_cca:
                period_str = 'period_' + str(period)

                for (k, xy_pair) in current.items():
                    views_str = 'views_' + '-'.join([str(j) for j in k])
                    path = '_'.join(
                        [period_str, views_str, 'dtcwt_heat_matrix.thang'])

                    for (l, mat) in xy_pair.items():
                        if self.cca_dir is not None:
                            current_path = os.path.join(
                                self.cca_dir, l + '_' + path)

                        with open(current_path, 'w') as f:
                            np.save(f, mat)

        # TODO: Do multi-view CCA on magnitude of coefficients

        # TODO: Maybe also do pairwise and multi-view on phase
            
        # TODO: Do CCA on unmodified complex coefficients (probably not)

    def _compute_correlation(self):

        for (period, (Yhs, Yls)) in enumerate(self.wavelets):
            correlation = self._get_period_correlation(
                Yhs, Yls)

            self.correlation.append(correlation)

            if self.save_correlation:
                for (k, hm) in correlation.items():
                    period_str = 'period_' + str(period)
                    views_str = 'views_' + '-'.join([str(i) for i in k])
                    path = '_'.join(
                        [period_str, views_str, 'dtcwt_heat_matrix.thang'])

                    if self.correlation_dir is not None:
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

        for i in xrange(self.num_views):
            for j in xrange(i, self.num_views):
                correlation.insert(i, j, get_matrix(i, j))

        return correlation

    def _get_wavelet_transforms(self):

        data = [ds.get_data() for ds in self.servers]
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
            print [d.shape for d in current_data]
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

                """
                (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                    view, 
                    int(log(view.shape[0], 2)) - 2,
                    self.biorthogonal, 
                    self.qshift)

                Yls[i].append(Yl)
                Yhs[i].append(Yh)
                """

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
            """
            resampled.append(_get_resampled_view(ds, rate))
            """
        for process in processes:
            resampled.append(process.get())

        return resampled

    def _show_cca(self):

        timelines = SPUD(
            self.num_views, default=list, no_double=True)

        for (i, period) in enumerate(self.pairwise_cca):
            for (k, xy_pair) in period.items():
                timelines.get(k[0], k[1]).append(xy_pair)

        for (k, l) in timelines.items():
            self._plot_cca(k, l)

    def _show_correlation(self):

        timelines = SPUD(self.num_views, default=list)
        prev = None

        for (i, period) in enumerate(self.correlation):
            for (k, hm) in period.items():
                """
                if i > 0:
                    timelines[k].append(hm - prev[k])
                """

                timelines.get(k[0], k[1]).append(hm)

            prev = period

        for (k, l) in timelines.items():
            self._plot_correlation(k, l)

    def _plot_cca(self, key, timeline):

        (i, j) = key
        (nx, px) = timeline[0]['Xw'].shape
        (ny, py) = timeline[0]['Yw'].shape
        names = [ds.name() for ds in self.servers]
        title = 'CCA decomposition of views ' + \
            names[i] + ' and ' + names[j] + \
            ' by decimation level'
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
            width=100*X_t.shape[1],
            height=50*X_t.shape[0],
            pos_color_scheme=X_pos_color_scheme,
            neg_color_scheme=X_neg_color_scheme)
        Y_plot = plot_matrix_heat(
            Y_t,
            x_labels,
            yy_labels,
            title,
            x_name,
            y_name,
            val_name,
            width=100*X_t.shape[1],
            height=50*X_t.shape[0])

        plot = Column(*[X_plot, Y_plot])
        filename = get_ts(
            'cca_of_wavelet_coefficients_' +
            names[i] + '_' + names[j]) + '.html'
        filepath = os.path.join(self.plot_path, filename)

        output_file(
            filepath, 
            'cca_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        #show(plot)

    def _plot_correlation(self, key, timeline):

        (i, j) = key
        (n, p) = timeline[0].shape
        names = [ds.name() for ds in self.servers]
        title = 'Correlation of views ' + names[i] + ' and ' + names[j] + ' by decimation level'
        x_name = 'decimation level'
        y_name = 'decimation level'
        x_labels = ['2^' + str(-k) for k in xrange(p)]
        y_labels = ['2^' + str(-k) for k in xrange(n)]
        val_name = 'correlation'
        plots = []

        for (k, hm) in enumerate(timeline):
            pos_color_scheme = None
            neg_color_scheme = None

            """
            if k % 2 > 0:
                pos_color_scheme = list(reversed(BuPu9))
                neg_color_scheme = list(reversed(Oranges9))
            """

            hmp = plot_matrix_heat(
                hm,
                x_labels,
                y_labels,
                title,
                x_name,
                y_name,
                val_name,
                pos_color_scheme=pos_color_scheme,
                neg_color_scheme=neg_color_scheme)

            plots.append(hmp)

        plot = Column(*plots)
        filename = get_ts(
            'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j]) + '.html'
        filepath = os.path.join(self.plot_path, filename)

        output_file(
            filepath, 
            'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        #show(plot)

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

def _get_dt_index(rows, f, dt):

    start = mktime(dt.timetuple())
    times = (np.arange(rows) * f + start).tolist()

    return [datetime.fromtimestamp(t)
            for t in times]
