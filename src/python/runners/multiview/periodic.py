import numpy as np
import spancca as scca

from wavelets import dtcwt
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9
from sklearn.cross_decomposition import CCA
from math import log

class MVCCADTCWTRunner:

    def __init__(self, 
        biorthogonal,
        qshift,
        servers, 
        period):

        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.servers = servers
        self.period = period

        self.num_views = len(self.servers)
        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        (Yls, Yhs) = self._get_wavelet_transforms()

        # Get heat plots
        wavelet_matrices = []
        heat_matrices = [] 
        heat_plots = []

        for period in xrange(len(Yhs[0])):
            Yhs_period = [view[period] for view in Yhs]
            Yls_period = [view[period] for view in Yls]
            Ys_matrices = [self._trunc_and_concat(Yh_p, Yl_p)
                            for (Yh_p, Yl_p) in zip(Yhs_period, Yls_period)]

            wavelet_matrices.append(Ys_matrices)
            heat_matrices.append(
                self._get_heat_matrices(Ys_matrices))
            heat_plots.append(
                {k : self._get_matrix_heat_plots(hm, k)
                 for (k, hm) in heat_matrices[-1].items()})

        # TODO: so many heat plots; need to design a clean way to display all of them

        # What exactly am I doing CCA on? Right, the matrices of coefficients
        # What is the output like for the 2d wavelets though. Can I still do standard CCA?

        # TODO: Do (multi-view?) CCA on magnitude of coefficients

        # TODO: Do CCA on np.real(coefficients)

        # TODO: Do CCA on np.imag(coefficients)
        
        # TODO: Do CCA on unmodified complex coefficients

        return heat_plots

    def _get_wavelet_transforms(self):

        data = [ds.get_data() for ds in self.servers]
        min_length = min(
            [ds.rows() for ds in self.servers])
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        k = 0

        while (k + 1) * self.period < min_length:
            begin = k * self.period
            end = begin + self.period
            current_data = [view[begin:end,:] for view in data]

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

    def _get_heat_matrices(self, Yhs_matrices):

        get_matrix = lambda i,j: np.dot(
            Yhs_matrices[i].T, Yhs_matrices[j])

        return {frozenset([i,j]): get_matrix(i,j)
                for i in xrange(self.num_views - 1)
                for j in xrange(i + 1, self.num_views)}

    def _get_matrix_heat_plots(self, heat_matrix, key):

        # TODO: figure out how to get ordered key
        print key
        (i,j) = tuple(key)
        names = [ds.name() for ds in self.servers]
        (n, p) = heat_matrix.shape
        x_labels = ['2^' + str(-i) for k in xrange(p)]
        y_labels = ['2^' + str(-i) for k in xrange(n)]
        title = 'Correlation of views ' + names[i] + ' and ' + names[j] + '  by decimation level'
        x_name = 'decimation level'
        y_name = 'decimation level'
        val_name = 'correlation'
        ps = plot_matrix_heat(
            heat_matrix,
            x_labels,
            y_labels,
            title,
            x_name,
            y_name,
            val_name)

        return ps

    def _trunc_and_concat(self, Yh, Yl):

        hi_and_lo = Yh + [Yl]
        min_length = min(
            [item.shape[0] for item in hi_and_lo])
        truncd = [item[:min_length,:] for item in hi_and_lo]

        return np.hstack(truncd)
