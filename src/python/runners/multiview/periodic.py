import os

import numpy as np
import pandas as pd

from drrobert.file_io import get_timestamped as get_ts
from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9
from bokeh.plotting import output_file
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

        for period in self.heat_matrices:
            for (k, hm) in period:
                self._plot_matrix_heat(hm, k)

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
            period = int(info[1])
            views = frozenset(
                [int(i) for i in info[3].split('-')]) 

            self.heat_matrices[period][views] = hm

    def _compute_heat_matrices(self):

        (Yls, Yhs) = self._get_wavelet_transforms()

        for period in xrange(len(Yhs[0])):
            Yhs_period = [view[period] for view in Yhs]
            Yls_period = [view[period] for view in Yls]
            Ys_matrices = [self._fill_and_concat(Yh_p, Yl_p)
                            for (Yh_p, Yl_p) in zip(Yhs_period, Yls_period)]
            period_heat = self._get_heat_matrices(Ys_matrices)

            self.wavelet_matrices.append(Ys_matrices)
            self.heat_matrices.append(period_heat)

            if self.save_heat:
                print 'Saving heat matrices for period', period
                for (k, hm) in period_heat.items():
                    period_str = 'period_' + str(period)
                    views_str = 'views_' + '-'.join([str(i) for i in k])
                    path = '_'.join(
                        [period_str, views_str, 'dtcwt_heat_matrix.thang'])

                    if self.heat_dir is not None:
                        path = os.path.join(self.heat_dir, path)

                    with open(path, 'w') as f:
                        print 'Saving heat matrix for view pair', k
                        np.save(f, hm)

    def _get_wavelet_transforms(self):

        data = self._get_resampled_data()
        factor = self.period * max(self.rates)
        min_length = min(
            [ds.rows() for ds in self.servers])
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        k = 0

        while (k + 1) * factor < min_length:
            current_data = [view[k * factor: (k+1) * factor,:] 
                            for view in data]

            for (i, view) in enumerate(current_data):
                (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                    view, 
                    int(log(view.shape[0], 2)) - 1,
                    self.biorthogonal, 
                    self.qshift)
                     
                Yls[i].append(Yl)
                Yhs[i].append(Yh)

            k += 1

        return (Yls, Yhs)

    def _get_resampled_data(self):

        data = [ds.get_data() for ds in self.servers]
        loaders = [ds.get_status()['data_loader'] 
                   for ds in self.servers]
        dts = [l.get_status()['start_times'][0]
               for l in loaders]
        freqs = [1.0 / r for r in self.rates]
        dt_indexes = [pd.DatetimeIndex(
                        data=self._get_dt_index(ds.rows(), f, dt))
                      for (dt, f, ds) in zip(dts, freqs, self.servers)]
        series = [pd.Series(data=view[:,0], index=dti) 
                  for (dti, view) in zip(dt_indexes, data)]

        return [s.resample('L').pad().as_matrix()
                for s in series]

    def _get_dt_index(self, rows, f, dt):

        start = mktime(dt.timetuple())
        times = (np.arange(rows) * f + start).tolist()

        return [datetime.fromtimestamp(t)
                for t in times]

    def _get_heat_matrices(self, Yhs_matrices):

        get_matrix = lambda i,j: np.dot(
            Yhs_matrices[i].T, Yhs_matrices[j])

        return {frozenset([i,j]): get_matrix(i, j)
                for i in xrange(self.num_views)
                for j in xrange(i, self.num_views)}

    def _plot_matrix_heat(self, heat_matrix, key):

        # TODO: figure out how to get ordered key
        (i,j) = tuple(key)
        names = [ds.name() for ds in self.servers]
        (n, p) = heat_matrix.shape
        x_labels = ['2^' + str(-k) for k in xrange(p)]
        y_labels = ['2^' + str(-k) for k in xrange(n)]
        title = 'Correlation of views ' + names[i] + ' and ' + names[j] + '  by decimation level'
        x_name = 'decimation level'
        y_name = 'decimation level'
        val_name = 'correlation'
        p = plot_matrix_heat(
            heat_matrix,
            x_labels,
            y_labels,
            title,
            x_name,
            y_name,
            val_name)

        filename = get_ts(
            'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j]) + '.html'
        filepath = os.path.join(self.plot_path, filename)
        output_file(filepath, 'correlation_of_wavelet_coefficients_' +
            names[i] + '_' + names[j])

        show(p)

    def _fill_and_concat(self, Yh, Yl):

        hi_and_lo = Yh + [Yl]
        filled = np.zeros((Yh.shape[0], len(hi_and_lo))) 

        for (i, y) in enumerate(hi_and_lo):
            filled[::2**i,i] = np.copy(y)

        return filled
