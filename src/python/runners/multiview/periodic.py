import numpy as np
import spancca as scca

from wavelets import dtcwt
from lazyprojector import plot_matrix_heat as plot_mh
from bokeh.palettes import BuPu9
from sklearn.cross_decomposition import CCA

class MVCCADTCWTRunner:

    def __init__(self, 
        biorthogonal,
        qshift,
        nlevels,
        servers, 
        period):

        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.nlevels = nlevels
        self.servers = servers
        self.period = period

        self.num_views = len(self.servers)
        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        # TODO: understand the meaning behind Yls and Yhs
        # TODO: figure out what to do with Yls
        (Yls, Yhs) = self._get_wavelet_transforms()

        # Get heat plots
        wavelet_matrices = []
        heat_matrices = [] 
        heat_plots = []

        for period in xrange(len(Yhs[0])):
            Yhs_period = [view[period] for view in Yhs]
            Yhs_matrices = [self._trunc_and_concat(Yh, Yl)
                            for Yh_p in Yhs_period]

            wavelet_matrices.append(Yhs_matrices)
            heat_matrices.append(
                self._get_heat_matrices(Yhs_matrices))
            heat_plots.append(
                {k : self._get_matrix_heat_plots(hm)
                 for (k, hm) in heat_matrices.items()})

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
        print min_length
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        k = 0

        while (k + 1) * self.period < min_length:
            begin = k * self.period
            end = begin + self.period
            current_data = [view[begin:end,:] for view in data]

            for (i, view) in enumerate(current_data):
                # TODO: test twod wavelet encoding stuff
                (Yl, Yh, _) = dtcwt.twod.dtwavexfm2(
                    view, 
                    self.nlevels, 
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

    def _get_matrix_heat_plots(self, heat_matrix, i, j):

        (n, p) = heat_matrix.shape
        x_labels = ['2^' + str(-i) for i in xrange(p)]
        y_labels = ['2^' + str(-i) for i in xrange(n)]
        title = 'Correlation of two views by decimation level'
        x_name = str(j) + ' decimation level'
        y_name = str(i) + ' decimation level'
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
