import os
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.animation import AVConvWriter
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from drrobert.stats import get_pearson_matrix as get_pm
from lazyprojector import plot_matrix_heat

class ViewPairwiseCorrelation:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        show=False):

        self.show = show

        self._init_dirs(
            show, 
            save_load_dir)

        self.servers = servers
        self.num_subperiods = num_subperiods
	self.subjects = self.servers.keys()
        self.names = [s.get_status()['data_loader'].name()
                      for s in self.servers.values()[0]]
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = len(self.servers.values()[0])
        self.num_periods = {s : int(servers[0].num_periods / self.num_subperiods)
                            for (s, servers) in self.servers.items()}
	self.max_periods = max(self.num_periods.values())

        self.correlation = {s : SPUD(self.num_views)
                            for s in self.subjects}

    def run(self):

        if self.show:
            self._load()
            self._show()
        else:
            self._compute()

    def _init_dirs(self, 
        show, 
        save_load_dir):

        if show:
            self.save_load_dir = save_load_dir
        else:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)

        hdf5_path = os.path.join(
            self.save_load_dir,
            'correlation.hdf5')
        self.hdf5_repo = h5py.File(
            hdf5_path, 'w' if not show else 'r')
        self.plot_dir = init_dir(
            'plots',
            show,
            self.save_load_dir) 
        self.full_time_dir = init_dir(
            'full-time',
            show,
            self.plot_dir)

    def _compute(self):

        for (s, servers) in self.servers.items():
            print 'Computing correlations for subject', s
            spud = self.correlation[s]

            for sp in xrange(self.num_subperiods * self.num_periods[s]):
                subperiods = [s.get_data() for s in servers]

                for k in spud.keys():
                    v1_mat = subperiods[k[0]]
                    v2_mat = subperiods[k[1]]
                    (n1, p1) = v1_mat.shape
                    (n2, p2) = v2_mat.shape

                    if n1 < n2:
                        num_reps = int(float(n2) / n1)
                        repped = np.zeros((n2, p1), dtype=complex)
                        
                        for r in xrange(num_reps):
                            max_len = repped[r::num_reps,:].shape[0]
                            repped[r::num_reps,:] = np.copy(
                                v1_mat[:max_len,:])

                        v1_mat = repped

                    elif n2 < n1:
                        num_reps = int(float(n1) / n2)
                        repped = np.zeros((n1, p2), dtype=complex)
                        
                        for r in xrange(num_reps):
                            max_len = repped[r::num_reps,:].shape[0]
                            repped[r::num_reps,:] = np.copy(
                                v2_mat[:max_len,:])

                        v2_mat = repped
                    
                    correlation = get_pm(
                        np.absolute(v1_mat), 
                        np.absolute(v2_mat))

                    self._save(
                        correlation,
                        s,
                        k,
                        sp)
                     
    def _save(self, c, s, v, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_string = str(v[0]) + '-' + str(v[1])

        if v_string not in s_group:
            s_group.create_group(v_string)

        v_group = s_group[v_string]
        sp_string = str(sp)

        sp_group.create_dataset(sp_string, data=c)

    def _load(self):

        for (s, spud) in self.correlation.items():
            num_subperiods = self.num_subperiods * self.num_periods[s]

            for k in spud.keys():
                l = [None] * num_subperiods

                spud.insert(k[0], k[1], l)
                
        for s in self.subjects:
            s_group = self.hdf5_repo[s]

            for (k_str, k_group) in s_group.items():
                vs = [int(v) for v in k_str.split('-')]

                for (sp_str, corr) in k_group.items():
                    sp = int(sp_str)

                    corr = np.array(corr)
                    
                    self.correlation[s].get(vs[0], vs[1])[sp] = corr

    def _show(self):

        for (s, spud) in self.correlation.items():
            for ((v1, v2), periods) in spud.items():
                m_plot = self._plot_movie(s, v1, v2, periods)

    # TODO: consider getting rid of load and just working directly from hdf5 repo
    def _plot_movie(self, s, v1, v2, subperiods):

        # TODO: pick a good fps
        writer = AVConvWriter(fps=1)
        fig = plt.figure()
        get_plot = lambda c, sp, p: self._get_correlation_plot(
            c, p, sp, v1, v2)
        num_frames = self.num_periods[s] * self.num_subperiods
        filename = 'views_' + \
            self.names[v1] + \
            '-' + \
            self.names[v2] + \
            '.mp4'
        path = os.path.join(
            self.full_time_dir, filename)

        with writer.saving(fig, path, num_frames):
            for (sp, corr) in enumerate(subperiods):
                if sp % self.num_periods[s] == 0:
                    do_something = 'Poop'
                    # TODO: add frame to indicate end of period

                plot = get_plot(corr, sp, p)

                writer.grab_frame()

    def _get_correlation_plot(self, c, sp, v1, v2):

        (m, n) = c.shape
        x_labels = ['2^{:02d}'.format(i) 
                    for i in xrange(n)]
        y_labels = ['2^{:02d}'.format(i)
                    for i in xrange(m)]
        title = ' '.join([
            'Frequency component correlation of view',
            str(v1),
            'vs',
            str(v2),
            'for subperiod',
            str(sp)])
        x_name = 'Subsampling rate for view 1 (' + self.names[v2] + ')'
        y_name = 'Subsampling rate for view 2 (' + self.names[v1] + ')'
        val_name = 'Pearson correlation'

        return plot_matrix_heat(
            c,
            x_labels,
            y_labels,
            title,
            x_name,
            y_name,
            val_name,
            vmax=1,
            vmin=-1)
        
    def _save_and_clear_plot(self, plot): 

        plot.get_figure().savefig(
                path, format='pdf')
        sns.plt.clf()
