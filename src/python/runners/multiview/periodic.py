import numpy as np
import spancca as scca

from wavelets import dtcwt
from lazyprojector import plot_matrix_heat as plot_mh
from bokeh.palettes import BuPu9
from sklearn.cross_decomposition import CCA

class MultiviewDTCWTCCAAnalysisRunner:

    def __init__(self, 
        get_biort,
        get_qshift,
        nlevels,
        servers, 
        period, 
        max_iter):

        self.model = model
        self.servers = servers
        self.period = period
        self.max_iter = max_iter

        self.num_views = len(self.servers)
        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        # TODO: understand the meaning behind Yls and Yhs
        # TODO: figure out what to do with Yls
        (Yls, Yhs) = self._get_wavelet_transforms()

        # Get heat plots
        heat_matrices = [] 
        heat_plots = []

        for period in xrange(len(Yhs[0])):
            Yhs_period = [view[period] for view in Yhs]

            heat_matrices.append(
                self._get_heat_matrices(Yhs_period))
            heat_plots.append(
                {k : self._get_matrix_heat_plots(hm)
                 for (k, hm) in heat_matrices.items()})

        # TODO: so many heat plots; need to design a clean way to display all of them

        # What exactly am I doing CCA on? Right, the matrices of coefficients
        # What is the output like for the 2d wavelets though. Can I still do standard CCA?

        # Do CCA on magnitude of coefficients

        # Do CCA on np.real(coefficients)

        # Do CCA on np.imag(coefficients)
        
        # Do CCA on unmodified complex coefficients

        return heat_plots

    def _get_wavelet_transforms(self):

        data = [ds.get_data() for ds in self.servers]
        min_length = min(
            [ds.rows() for ds in self.servers])
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        k = 0

        while (k + 1) * period < min_length:
            begin = k * period
            end = begin + period
            current_data = [view[begin:end,:] for view in data]

            for (i, view) in enumerate(current_data):
                # TODO: test twod wavelet encoding stuff
                (Yl, Yh, _) = dtcwt.twod.dtwavexfm(
                    view, 
                    self.nlevels, 
                    self.get_biort, 
                    self.get_qshift)
                     
                Yls[i].append(Yl)
                Yhs[i].append(Yh)

            k += 1

        return (Yls, Yhs)

    def _get_heat_matrices(self, Yhs):

        heat_matrices = [frozenset([i,j]):
                         None
                         for i in xrange(self.num_views) 
                         for j in xrange(i, self.num_views-1)]

        for i in xrange(self.num_views):
            Yh_i = Yhs[i]

            for j in xrange(i, self.num_views-1):
                Yh_j = Yhs[j]

                # Make the heat matrix for Yh_i vs Yh_j
                min_length = min(
                    [item.size[0] for item in Yh_j] +
                    [item.size[0] for item in Yh_i])

                Yh_i_matrix = self._trunc_and_concat(
                    Yh_i, min_length)
                Yh_j_matrix = self._trunc_and_concat(
                    Yh_j, min_length)
                ij_heat_matrix = np.dot(Yh_i_matrix.T, Yh_j_matrix)

                heat_matrices[frozenset([i,j])] = ij_heat_matrix

        return heat_matrices

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

    def _trunc_and_concat(self, Yh, min_length):

        truncd = [item[:min_length,:] for item in Yh]

        return np.hstack(truncd)
