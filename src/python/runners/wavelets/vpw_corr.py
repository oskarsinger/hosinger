import os
import matplotlib
import h5py

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import utils as rwu
import matplotlib.pyplot as plt

from matplotlib.animation import AVConvWriter
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.arithmetic import get_running_avg
from drrobert.file_io import get_timestamped as get_ts
from drrobert.stats import get_pearson_matrix as get_pm
from lazyprojector import plot_matrix_heat
from data.loaders.e4.utils import get_symptom_status

class ViewPairwiseCorrelationRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        save=False,
        load=False,
        show=False):

        self.save = save
        self.load = load
        self.show = show

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.num_subperiods = dtcwt_runner.num_sps
	self.subjects = self.wavelets.keys()
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.num_periods = {s : len(self.wavelets[s])
			    for s in self.subjects}
	self.max_periods = max(self.num_periods.values())

        self.correlation = {s : SPUD(self.num_views)
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show()

    def _init_dirs(self, 
        save, 
        load, 
        show, 
        save_load_dir):

        if (show or save) and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        hdf5_path = os.path.join(
            self.save_load_dir,
            'correlation.hdf5')
        self.hdf5_repo = h5py.File(
            hdf5_path, 'w' if save else 'r')
        self.plot_dir = rwu.init_dir(
            'plots',
            show,
            self.save_load_dir) 
        self.full_time_dir = rwu.init_dir(
            'full-time',
            show,
            self.plot_dir)

        """
        subperiod_dir = rwu.init_dir(
            'subperiod',
            show,
            self.plot_dir)

        self.subperiod_dirs = []

        for i in xrange(self.num_subperiods):
            sp_i_dir = rwu.init_dir(
                'sp-' + str(i),
                show,
                subperiod_dir)

            self.subperiod_dirs.append(sp_i_dir)
        """

    def _compute(self):

        for (s, s_wavelets) in self.wavelets.items():
            print 'Computing correlations for subject', s
            spud = self.correlation[s]

            for (p, day) in enumerate(s_wavelets):
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) =  subperiod[k[0]]
                        (Yh2, Yl2) =  subperiod[k[1]]
                        Y1_mat = rwu.get_padded_wavelets(Yh1, Yl1)
                        Y2_mat = rwu.get_padded_wavelets(Yh2, Yl2)
                        (n1, p1) = Y1_mat.shape
                        (n2, p2) = Y2_mat.shape

                        if n1 < n2:
                            num_reps = int(float(n2) / n1)
                            repped = np.zeros((n2, p1), dtype=complex)
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    Y1_mat[:max_len,:])

                            Y1_mat = repped

                        elif n2 < n1:
                            num_reps = int(float(n1) / n2)
                            repped = np.zeros((n1, p2), dtype=complex)
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    Y2_mat[:max_len,:])

                            Y2_mat = repped
                        
                        correlation = get_pm(
                            np.absolute(Y1_mat), 
                            np.absolute(Y2_mat))

                        self._save(
                            correlation,
                            s,
                            k,
                            p,
                            sp)
                     
    def _save(self, c, s, v, p, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_string = str(v[0]) + '-' + str(v[1])

        if v_string not in s_group:
            s_group.create_group(v_string)

        v_group = s_group[v_string]
        p_string = str(p)

        if p_string not in v_group:
            v_group.create_group(p_string)

        p_group = v_group[p_string]
        sp_string = str(sp)

        p_group.create_dataset(sp_string, data=c)

    def _load(self):

        for (s, spud) in self.correlation.items():
            for k in spud.keys():
                l_of_l = [[None] * self.num_subperiods
                          for i in xrange(self.num_periods[s])]

                spud.insert(k[0], k[1], l_of_l)
                
        for s in self.subjects:
            s_group = self.hdf5_repo[s]

            for (k_str, sp_group) in s_group.items():
                vs = [int(v) for v in k_str.split('-')]

                for (p_str, p_group) in sp_group.items():
                    p = int(p_str)

                    for (sp_str, corr) in p_group.items():
                        sp = int(sp_str)
                        corr = np.array(corr)
                        
            	        self.correlation[s].get(vs[0], vs[1])[p][sp] = corr

    def _show(self):

        for (s, spud) in self.correlation.items():
            for ((v1, v2), periods) in spud.items():
                m_plot = self._get_movie_plot(s, v1, v2, periods)

    # TODO: consider getting rid of load and just working directly from hdf5 repo
    def _get_movie_plot(self, s, v1, v2, periods):

        # TODO: pick a good fps
        writer = AVConvWriter(fps=1)
        fig = plt.figure()
        get_plot = lambda c, a, sp, p: self._get_correlation_plot(
            c, v1, v2, p, sp, a)
        num_frames = self.num_periods[s] * self.num_subperiods
        filename = 'views_' + \
            self.names[v1] + \
            '-' + \
            self.names[v2] + \
            '.mp4'
        path = os.path.join(
            self.full_time_dir, filename)
        print 'path', path

        with writer.saving(fig, path, num_frames):
            for (p, subperiods) in enumerate(periods):
                # TODO: add frame to indicate end of 24-hour period

                for (sp, corr) in enumerate(subperiods):
                    plot = get_plot(corr, fig.axes(), sp, p)

                    writer.grab_frame()

    def _get_correlation_plot(self, c, p, sp, v1, v2, ax):

        (m, n) = corr.shape
        x_labels = ['2^{:02i}'.format(i) 
                    for i in xrange(n)]
        y_labels = ['2^{:02i}'.format(i)
                    for i in xrange(m)]
        title = ' '.join([
            'Frequency component correlation of view',
            str(v1),
            'vs',
            str(v2),
            'for subperiod',
            str(sp),
            'of period',
            str(p)])
        x_name = 'Subsampling rate for view ' + self.names[v2]
        y_name = 'Subsampling rate for view ' + self.names[v1]
        val_name = 'Pearson correlation'

        fn = '_'.join(title.split()) + '.pdf'
        path = os.path.join(self.plot_dir, fn)

        return plot_matrix_heat(
            corr,
            x_labels,
            y_labels,
            title,
            x_name,
            y_name,
            val_name,
            vmax=1,
            vmin=-1,
            ax=ax)
        
    def _save_and_clear_plot(self, plot): 
        plot.get_figure().savefig(
                path, format='pdf')
        sns.plt.clf()
