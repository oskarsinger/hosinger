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

class NTPFViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        clock_time=False,
        show=False):

        self.servers = servers
        self.show = show
        self.clock_time = clock_time
        self.subjects = self.servers.keys()
        self.num_subperiods = num_subperiods
        self.subperiod = int(24.0 * 3600.0 / self.num_subperiods)
        self.loaders = {s : [ds.get_status()['data_loader'] for ds in dss]
                        for (s, dss) in self.servers.items()}
        self.names = {s : [dl.name() for dl in dls]
                      for (s, dls) in self.loaders.items()}
        self.num_views = len(self.servers.values()[0])
        self.num_periods = {s : int(servers[0].num_batches / self.num_subperiods)
                            for (s, servers) in self.servers.items()}
        self.cca_names = [
            'n_time_p_frequency',
            'n_time_p_frequency_cc']

        self._init_dirs(
            show, 
            save_load_dir)

        get_container = lambda: {s : SPUD(
                                    self.num_views, 
                                    no_double=True)
                                 for s in self.subjects}
        self.ccas = {n : get_container()
                     for n in self.cca_names}

    def run(self):

        if self.show:
            self._load()

            self._show_n_time_p_frequency()
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

            model_dir = get_ts('VPWCCA')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)

        get_path = lambda n: os.path.join(
            self.save_load_dir, n)
        hdf5_paths = {n : get_path(n) for n in self.cca_names}
        self.hdf5_repos = {n : h5py.File(p, 'r' if show else 'w')
                           for (n, p) in hdf5_paths.items()}
        self.plot_dir = init_dir(
            'plots',
            show,
            self.save_load_dir) 

    def _compute(self):

        for (s, servers) in self.servers.items():
            print 'Computing CCAs for subject', s

            for sp in xrange(self.num_subperiods * self.num_periods[s]):
                subperiods = [ds.get_data() for ds in servers]

                for v1 in xrange(self.num_views):
                    for v2 in xrange(i+1, self.num_views):
                        v1_mat = subperiods[v1]
                        v2_mat = subperiods[v2]
                        (v1_mat, v2_mat) = get_matched_dims(
                            v1_mat, v2_mat)
                        ntpf = get_cca_vecs(v1_mat, v2_mat)
                        v1_cc = np.dot(v1_mat, ntpf[0])
                        v2_cc = np.dot(v2_mat, ntpf[1])
                        ntpfcc = v1_cc * v2_cc

                        self._save(
                            ntpf,
                            ntpfcc,
                            s,
                            v1,
                            v2,
                            sp)

    def _save(self, ntpf, ntpfcc, s, v1, v2, sp):

        if s not in self.hdf5_repo:
            repo.create_group(s)

        s_group = repo[s]
        v_str = str(v1) + '-' + str(v2)
        
        if v_str not in s_group:
            s_group.create_group(v_str)

        v_group = s_group[v_str]
        sp_str = str(sp)

        if sp_str not in v_group:
            v_group.create_group(sp_str)

        sp_group = v_group[sp_str]

        sp_group.create_dataset('Phi1', data=ntpf[0])
        sp_group.create_dataset('Phi2', data=ntpf[1])
        sp_group.create_dataset('CC', data=ntpfcc)

    def _load(self):

        for (s, spud) in self.cca.items():
            for k in spud.keys():
                l = [None] * self.num_subperiods * self.num_periods[s]

                spud.insert(k[0], k[1], l)
        
        for (s, s_group) in self.hdf5_repo.items():
            cca_s = cca[s]

            for (v_str, v_group) in s_group.items():
                (v1, v2) = [int(v) for v in v_str.split('-')]

                cca_vs = cca_s.get(v1, v2)

                for (sp_str, sp_group) in v_group.items():
                    sp = int(sp_str)
                    ntpf = (sp_group['Phi1'], sp_group['Phi2'])
                    ntpfcc = sp_group['CC']
                    
                    cca_vs[sp] = (ntpf, ntpfcc)

    def _show(self):

        default = lambda: {s : None for s in ntpfcc.keys()}
        tl_spuds = SPUD(
            self.num_views, 
            default=default, 
            no_double=True)

        for (s, spud) in ntpfcc.items():
            for ((v1, v2), subperiods) in spud.items():
                tl = np.hstack(subperiods)

                tl_spuds.get(v1, v2)[s] = tl

        for (s, spud) in self.ccas.items():
            
            print 'Generating n_time_p_frequency plots for', s

            for ((v1, v2), subperiods) in spud.items():

                print '\tGenerating plots for views', v1, v2

                fig = plt.figure()
                filename = '_'.join([
                    'subject', s,
                    '_views_',
                    self.names[s][v1] + '-' + self.names[s][v2]]) + '.png'
                path = os.path.join(
                    self.n_time_p_frequency_dir, filename)
                (nptf, ntpfcc) = unzip(subperiods)
                (Phi1s, Phi2s) = unzip(ntpf)
                title = 'View-pairwise cca (n time p frequency) for views ' + \
                    self.names[s][v1] + ' ' + self.names[s][v2] + \
                    ' of subject ' + s
                x_name = 'subperiod'
                y_name = 'dimension'
                v_name = 'canonical vector value'

                ax1 = fig.add_subplot(311)

                self._plot_matrix_heat(
                    s,
                    v1,
                    Phi1s,
                    x_name,
                    y_name,
                    v_name,
                    ax1)

                ax2 = fig.add_subplot(312)

                self._plot_matrix_heat(
                    s,
                    v2,
                    Phi2s,
                    x_name,
                    y_name,
                    v_name,
                    ax2)

                x_name = 'time'
                y_name = 'canonical correlation'
                ax3 = fig.add_subplot(313)

                self._plot_line(
                    s, 
                    v1, 
                    ntpfcc, 
                    'time', 
                    'canonical correlation', 
                    ax3)

                fig.suptitle(title)

                fn = '_'.join(title.split()) + '.png'
                path = os.path.join(
                    self.plot_dir, fn)

                fig.savefig(path, format='png')
                plt.clf()

    def _plot_matrix_heat(self, s, v, ccal, x_name, y_name, v_name, ax):

        tl = np.hstack(ccal)
        (n, m) = tl.shape
        (yl, xl) = (np.arange(n), np.arange(m))

        plot_matrix_heat(
            tl,
            xl,
            yl,
            '',
            x_name,
            y_name,
            v_name,
            vmax=1,
            vmin=-1,
            ax=ax)

        if self.clock_time:
            start_time = self.loaders[s][v].get_status()['start_times'][0]
            factor =  float(self.subperiod) / n
            x_axis = np.array(get_dti(
                n,
                factor,
                start_time))

            plt.xticks(xl, x_axis)
            ax.xaxis.set_major_locator(
                mdates.HourLocator(interval=12))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%b %d %H:%M'))

    def _plot_line(self, s, v, datal, x_name, y_name, ax):

        tl = np.vstack(datal)
        n = tl.shape[0]
        x_axis = None

        if self.clock_time:
            start_time = self.loaders[s][v].get_status()['start_times'][0]
            num_sps = self.num_subperiods * self.num_periods[s]
            factor = num_sps * self.subperiod / n
            x_axis = np.array(get_dti(
                n,
                factor,
                start_time))
            ax.xaxis.set_major_locator(
                mdates.HourLocator(interval=12))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%b %d %H:%M'))
        else:
            x_axis = np.arange(n)[:,np.newaxis]

        plot = ax.plot(x_axis, tl)

        plot.xtitle(x_name)
        plot.y_title(y_name)
