import os
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.stats import get_cca_vecs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from drrobert.misc import unzip
from drrobert.ts import get_dt_index as get_dti
from exploratory.mvc.utils import get_matched_dims
from math import log, ceil

class NFPTViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        nnz=1,
        clock_time=False,
        show=False):

        self.servers = servers
        self.num_subperiods = num_subperiods
        self.nnz = nnz
        self.clock_time = clock_time
        self.show = show

    def run(self):

        if self.show:
            self._load()
            self._show()
        else:
            self._compute()

    def _compute(self):

        for (s, servers) in self.servers.items():
            print 'Computing CCAs for subject', s

            for sp in xrange(self.num_subperiods * self.num_periods[s]):
                subperiods = [ds.get_data() for ds in servers]

                for v1 in xrange(self.num_views):
                    for v2 in xrange(i+1, self.num_views):
                        v1_mat = subperiods[i]
                        v2_mat = subperiods[j]
                        (v1_mat, v2_mat) = get_matched_dims(
                            v1_mat, v2_mat)

                        n_frequency_p_time = get_cca_vecs(
                            v1_mat.T, v2_mat.T,
                            num_nonzero=self.nnz)

    def _show(self):

        tl_spuds = {s: SPUD(self.num_views, no_double=True)
                    for s in self.ccas.keys()}

        for (s, spud) in self.ccas[self.cca_names[2]].items():
            for ((v1, v2), subperiods) in spud.items():
                (phi1s, phi2s) = unzip(subperiods)
                tls = (
                    np.hstack(phi1s),
                    np.hstack(phi2s))

                tl_spuds[s].insert(v1, v2, tls)

        default = lambda: {}
        data_maps = SPUD(
            self.num_views,
            default=default,
            no_double=True)

        for (s, spud) in tl_spuds.items():
            for ((v1, v2), tl) in spud.items():
                s_key = 'Subject ' + s + ' view '
                factor = float(self.num_periods[s]) / tl.shape[0]
                # TODO: set up date axis
                x_axis = 'Something'
                phi1 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[0][:,np.newaxis],
                    None)
                phi2 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[1][:,np.newaxis],
                    None)
                data_maps.get(v1, v2)[s_key + str(1)] = phi1
                data_maps.get(v1, v2)[s_key + str(2)] = phi2

        fig = plt.figure()
        
        for ((v1, v2), dm) in data_maps.items():

            print 'Generating n_frequency_p_time_plots for', v1, v2

            x_name = 'time (days)'
            y_name = 'canonical vector value'
            title = 'View-pairwise canonical vectors' + \
                ' (n frequency p time) for views '

            for (i, (s, data)) in enumerate(dm.items()):

                print '\tGenerating plot for subject', s

                ax = fig.add_subplot(
                    len(self.subjects), 1, i+1)
                s_title = title + \
                    self.names[s][v1] + ' ' + self.names[s][v2]
                s_dm = {s : data}

                plot_lines(
                    s_dm,
                    x_name,
                    y_name,
                    s_title,
                    ax=ax)

            plt.clf()

