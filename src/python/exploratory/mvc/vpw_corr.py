import os
import json
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
from drrobert.ts import get_dt_index as get_dti
from lazyprojector import plot_matrix_heat

class ViewPairwiseCorrelation:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        clock_time=False,
        show=False):

        self.show = show

        self._init_dirs(
            show, 
            save_load_dir)

        self.servers = servers
        self.num_subperiods = num_subperiods
        self.clock_time = clock_time

	self.subjects = self.servers.keys()
        self.loaders = {s : [ds.get_status()['data_loader'] for ds in dss]
                        for (s, dss) in self.servers.items()}
        self.names = {s : [dl.name() for dl in dls]
                      for (s, dls) in self.loaders.items()}
        self.num_views = len(self.servers.values()[0])
        self.num_periods = {s : int(servers[0].num_batches / self.num_subperiods)
                            for (s, servers) in self.servers.items()}
	self.max_periods = max(self.num_periods.values())
        self.correlation = {s : SPUD(self.num_views, no_double=True)
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

            model_dir = get_ts('VPWC')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)

        hdf5_path = os.path.join(
            self.save_load_dir,
            'correlation.hdf5')
        self.hdf5_repo = h5py.File(
            hdf5_path, 'r' if show else 'w')
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

            for sp in xrange(self.num_subperiods * self.num_periods[s]):
                subperiods = [ds.get_data() for ds in servers]

                for i in xrange(self.num_views):
                    for j in xrange(i+1, self.num_views):
                        v1_mat = subperiods[i]
                        v2_mat = subperiods[j]
                        (n1, p1) = v1_mat.shape
                        (n2, p2) = v2_mat.shape

                        if n1 < n2:
                            num_reps = int(float(n2) / n1)
                            repped = np.zeros((n2, p1))
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    v1_mat[:max_len,:])

                            v1_mat = repped

                        elif n2 < n1:
                            num_reps = int(float(n1) / n2)
                            repped = np.zeros((n1, p2))
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    v2_mat[:max_len,:])

                            v2_mat = repped

                        correlation = get_pm(
                            v1_mat, 
                            v2_mat)

                        self._save(
                            correlation,
                            s,
                            i,
                            j,
                            sp)
                     
    def _save(self, c, s, v1, v2, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_string = str(v1) + '-' + str(v2)

        if v_string not in s_group:
            s_group.create_group(v_string)

        v_group = s_group[v_string]
        sp_string = str(sp)
        
        v_group.create_dataset(sp_string, data=c)

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
            print 'Generating plots for subject', s

            for ((v1, v2), subperiods) in spud.items():
                print '\tGenerating plot for views', v1, v2
                self._plot_movie(s, v1, v2, subperiods)

    # TODO: consider getting rid of load and just working directly from hdf5 repo
    def _plot_movie(self, s, v1, v2, subperiods):

        # TODO: pick a good fps
        writer = AVConvWriter(fps=0.25)
        fig = plt.figure()
        get_corr_plot = lambda c, sp, ax: self._get_correlation_plot(
            c, sp, v1, v2, s, ax)
        num_frames = self.num_periods[s] * self.num_subperiods
        filename = \
            'subject_' + s + \
            '_views_' + \
            self.names[s][v1] + \
            '-' + \
            self.names[s][v2] + \
            '.mp4'
        path = os.path.join(
            self.full_time_dir, filename)
        data1 = self.loaders[s][v1].get_data()
        data2 = self.loaders[s][v2].get_data()
        sp_length1 = int(data1.shape[0] / 
            (self.num_subperiods * self.num_periods[s]))
        sp_length2 = int(data2.shape[0] / 
            (self.num_subperiods * self.num_periods[s]))

        with writer.saving(fig, path, 100):
            for (sp, corr) in enumerate(subperiods):
                print '\t\tGenerating frame', sp
                ax1 = fig.add_subplot(311)

                print '\t\t\tGenerating corr plot'
                get_corr_plot(corr, sp, ax1)

                ax2 = fig.add_subplot(312)
                sp_data1 = data1[sp_length1 * sp:sp_length1 * (sp + 1),:]

                print '\t\t\tGenerating data plot 1'
                self._get_data_plot(
                    s, v1, sp, sp_data1, ax2)
                
                ax3 = fig.add_subplot(313)
                sp_data2 = data2[sp_length2 * sp:sp_length2 * (sp + 1),:]

                print '\t\t\tGenerating data plot 2'
                self._get_data_plot(
                    s, v2, sp, sp_data2, ax3)

                writer.grab_frame()
                plt.clf()

        for ds in self.servers[s]:
            ds.refresh()

        plt.close(fig)

    def _get_data_plot(self, s, v, sp, data, ax):

        x_axis = None

        if self.clock_time:
            dl = self.servers[s][v].get_status()['data_loader']
            start_time = self.loaders[s][v].get_status()['start_times'][0]
            n = data.shape[0]
            factor = 3600.0 / n
            x_axis = np.array(get_dti(
                n,
                factor,
                start_time,
                offset=3600.0 * (sp + 1)))[:,np.newaxis]
        else:
            x_axis = np.arange(data.shape[0])

        return ax.plot(x_axis, data)

    def _get_correlation_plot(self, c, sp, v1, v2, s, ax):

        (m, n) = c.shape
        x_labels = ['{:02d}'.format(i) 
                    for i in xrange(n)]
        y_labels = ['{:02d}'.format(i)
                    for i in xrange(m)]
        title = ' '.join([
            'Pearson correlation of view',
            self.names[s][v1],
            'vs',
            self.names[s][v2],
            'for subperiod',
            str(sp)])
        x_name = 'Dimensions of view 2: ' + self.names[s][v2]
        y_name = 'Dimensions of view 1: ' + self.names[s][v1]
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
            vmin=-1,
            ax=ax)
