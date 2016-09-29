import os

import numpy as np
import pandas as pd

from multiprocessing import Pool
from drrobert.file_io import get_timestamped as get_ts
from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9, Oranges9
from bokeh.plotting import output_file, show
from bokeh.models.layouts import Column
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
        heat_dir=None,
        load_heat=False,
        save_heat=False,
        show_plots=False,
        plot_path='.'):

        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.servers = servers
        self.period = period
        self.heat_dir = heat_dir
        self.load_heat = load_heat
        self.save_heat = save_heat
        self.show_plots = show_plots
        self.plot_path = plot_path

        self.rates = [ds.get_status()['data_loader'].get_status()['hertz']
                      for ds in self.servers]
        self.num_views = len(self.servers)
        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0
        self.wavelet_matrices = []
        self.heat_matrices = [] 

    def run(self):

        if self.load_heat:
            self._load_heat_matrices()
        else:
            self._compute_heat_matrices()

        if self.show_plots:
            self._show_plots()
        # What exactly am I doing CCA on? Right, the matrices of coefficients
        # What is the output like for the 2d wavelets though. Can I still do standard CCA?

        # TODO: Do (multi-view?) CCA on magnitude of coefficients

        # TODO: Do CCA on np.real(coefficients)

        # TODO: Do CCA on np.imag(coefficients)
        
        # TODO: Do CCA on unmodified complex coefficients

    def _show_plots(self):

        timelines = {k : [] for k in self.heat_matrices[0].keys()}
        prev = None

        for (i, period) in enumerate(self.heat_matrices):
            for (k, hm) in period.items():
                if i > 0:
                    timelines[k].append(hm - prev[k])

                timelines[k].append(hm)

            prev = period

        for (k, l) in timelines.items():
            self._plot_correlation_heat(k, l)

    def _load_heat_matrices(self):

        heat_matrices = {}

        for fn in os.listdir(self.heat_dir):
            path = os.path.join(self.heat_dir, fn)

            with open(path) as f:
                heat_matrices[fn] = np.load(f)

        num_periods = max(
            [int(k.split('_')[1]) 
             for k in heat_matrices.keys()])

        self.heat_matrices = [{} for i in xrange(num_periods)]

        for (k, hm) in heat_matrices.items():
            info = k.split('_')
            period = int(info[1]) - 1
            views = frozenset(
                [int(i) for i in info[3].split('-')]) 

            self.heat_matrices[period][views] = hm

    def _compute_heat_matrices(self):

        (Yls, Yhs) = self._get_wavelet_transforms()

        for period in xrange(len(Yhs[0])):
            print 'Computing heat matrices for period', period
            Yhs_period = [view[period] for view in Yhs]
            Yls_period = [view[period] for view in Yls]
            period_heat = self._get_period_heat_matrices(
                Yhs_period, Yls_period)

            self.wavelet_matrices.append((Yhs_period, Yls_period))
            self.heat_matrices.append(period_heat)

            if self.save_heat:
                for (k, hm) in period_heat.items():
                    period_str = 'period_' + str(period)
                    views_str = 'views_' + '-'.join([str(i) for i in k])
                    path = '_'.join(
                        [period_str, views_str, 'dtcwt_heat_matrix.thang'])

                    if self.heat_dir is not None:
                        path = os.path.join(self.heat_dir, path)

                    with open(path, 'w') as f:
                        np.save(f, hm)

    def _get_wavelet_transforms(self):

        # TODO: downsampled after wavelet coefficient
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

    def _get_period_heat_matrices(self, Yhs, Yls):

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

        return {frozenset([i,j]): get_matrix(i, j)
                for i in xrange(self.num_views)
                for j in xrange(i, self.num_views)}

    def _plot_correlation_heat(self, key, timeline):

        (i, j) = [None] * 2
        if len(key) == 1:
            (i, j) = list(key) * 2
        else:
            (i,j) = tuple(key)

        (n, p) = timeline[0].shape
        names = [ds.name() for ds in self.servers]
        title = 'Correlation of views ' + names[i] + ' and ' + names[j] + '\nby decimation level'
        x_name = 'decimation level'
        y_name = 'decimation level'
        x_labels = ['2^' + str(-k) for k in xrange(p)]
        y_labels = ['2^' + str(-k) for k in xrange(n)]
        val_name = 'correlation'
        plots = []

        for (k, hm) in enumerate(timeline):
            pos_color_scheme = None
            neg_color_scheme = None

            if k % 2 > 0:
                pos_color_scheme = list(reversed(BuPu9))
                neg_color_scheme = list(reversed(Oranges9))

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

        output_file(filepath, 'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])
        show(plot)

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
        new = y[::2**power,0]
        new_copy = np.copy(new)
        basis[:,i] = new_copy

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
