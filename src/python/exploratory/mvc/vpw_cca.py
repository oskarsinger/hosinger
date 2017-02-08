import os
import json
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.stats import get_cca_vecs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from drrobert.misc import unzip
from drrobert.ts import get_dt_index as get_dti
from lazyprojector import plot_matrix_heat, plot_lines
from math import log, ceil

class ViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        nnz=1,
        clock_time=False,
        show=False):

        self.servers = servers
        self.show = show
        self.nnz = nnz
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
	self.max_periods = max(self.num_periods.values())
        self.cca_names = [
            'n_time_p_frequency',
            'n_frequency_p_time',
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
            #self._show_n_frequency_p_time()

            self._show_n_time_p_frequency_cc()
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

        self.freqs_path = os.path.join(
            self.save_load_dir,
            'p_by_view.json')

        if show:
            with open(self.freqs_path) as f:
                line = f.readline()

                self.p_by_view = json.loads(line)
        else:
            self.p_by_view = [None] * self.num_views


        get_path = lambda n: os.path.join(
            self.save_load_dir, n)
        hdf5_paths = {n : get_path(n) for n in self.cca_names}
        self.hdf5_repos = {n : h5py.File(p, 'r' if show else 'w')
                           for (n, p) in hdf5_paths.items()}
        self.plot_dir = init_dir(
            'plots',
            show,
            self.save_load_dir) 
        self.n_time_p_frequency_dir = init_dir(
            self.cca_names[0],
            show,
            self.plot_dir)
        self.n_frequency_p_time_dir = init_dir(
            self.cca_names[1],
            show,
            self.plot_dir)
        self.n_time_p_frequency_cc_dir = init_dir(
            self.cca_names[2],
            show,
            self.plot_dir)

    def _compute(self):

        for (s, servers) in self.servers.items():
            print 'Computing CCAs for subject', s

            for sp in xrange(self.num_subperiods * self.num_periods[s]):
                subperiods = [ds.get_data() for ds in servers]

                for i in xrange(self.num_views):
                    v1_mat = subperiods[i]

                    for j in xrange(i+1, self.num_views):
                        v2_mat = subperiods[j]
                        n_time_p_frequency = get_cca_vecs(
                            v1_mat, v2_mat)
                        cca_dim = min(v1_mat.shape + v2_mat.shape)
                        n_frequency_p_time = get_cca_vecs(
                            v1_mat[:,:cca_dim].T,
                            v2_mat[:,:cca_dim].T,
                            num_nonzero=self.nnz)
                        n_time_p_frequency_cc = self._get_n_time_p_frequency_cc(
                            v1_mat,
                            v2_mat,
                            n_time_p_frequency)
                        stuff = {
                            self.cca_names[0]: n_time_p_frequency,
                            self.cca_names[1]: n_frequency_p_time,
                            self.cca_names[2]: n_time_p_frequency_cc}

                        self.p_by_view[i] = v1_mat.shape[1] 
                        self.p_by_view[j] = v2_mat.shape[1]

                        self._save(
                            stuff,
                            s,
                            i,
                            j,
                            sp)

        p_by_view_json = json.dumps(self.p_by_view)

        with open(self.freqs_path, 'w') as f:
            f.write(p_by_view_json)

    def _get_n_time_p_frequency_cc(self, v1_mat, v2_mat, n_time_p_frequency):

        v1_cc = np.dot(
            v1_mat, 
            n_time_p_frequency[0])
        v2_cc = np.dot(
            v2_mat, 
            n_time_p_frequency[1])

        return v1_cc * v2_cc

    def _save(self, cs, s, i, j, sp):

        for (n, repo) in self.hdf5_repos.items():
            if s not in repo:
                repo.create_group(s)

            s_group = repo[s]
            v_str = str(i) + '-' + str(j)
            
            if v_str not in s_group:
                s_group.create_group(v_str)

            v_group = s_group[v_str]
            sp_str = str(sp)

            if n in self.cca_names[:2]:
                if sp_str not in v_group:
                    v_group.create_group(sp_str)

                sp_group = v_group[sp_str]
                (Phi1, Phi2) = cs[n]

                sp_group.create_dataset('1', data=Phi1)
                sp_group.create_dataset('2', data=Phi2)
            else:
                v_group.create_dataset(sp_str, data=cs[n])

    def _load(self):

        for (n, n_repo) in self.hdf5_repos.items():
            cca = self.ccas[n]
            for (s, spud) in cca.items():
                for k in spud.keys():
                    l = [None] * self.num_subperiods * self.num_periods[s]

                    spud.insert(k[0], k[1], l)
            
            for (s, s_group) in n_repo.items():
                cca_s = cca[s]

                for (v_str, v_group) in s_group.items():
                    (v1, v2) = [int(v) for v in v_str.split('-')]

                    if not v1 == v2:
                        cca_vs = cca_s.get(v1, v2)

                        for (sp_str, sp_group) in v_group.items():
                            sp = int(sp_str)

                            if n in self.cca_names[:2]:
                                cca_vs[sp] = (
                                    np.array(sp_group['1']),
                                    np.array(sp_group['2']))
                            else:
                                cca_vs[sp] = np.array(sp_group)

    def _show_n_frequency_p_time(self):

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

    def _show_n_time_p_frequency_cc(self):

        ntpfcc = self.ccas[self.cca_names[2]]
        tl_spuds = {s: SPUD(self.num_views, no_double=True)
                    for s in ntpfcc.keys()}

        for (s, spud) in ntpfcc.items():
            for ((v1, v2), subperiods) in spud.items():
                tl = np.hstack(subperiods)

                tl_spuds[s].insert(v1, v2, tl)

        default = lambda: {s: None for s in ntpfcc.keys()}
        data_maps = SPUD(
            self.num_views,
            default=default,
            no_double=True)

        for (s, spud) in tl_spuds.items():
            for (k, tl) in spud.items():
                data = (
                    np.arange(len(tl))[:,np.newaxis], 
                    np.array(tl),
                    None)
                data_maps.get(k[0], k[1])[s] = data

        fig = plt.figure()

        for ((v1, v2), dm) in data_maps.items():
            
            print 'Generating n_time_p_frequency_plots_cc for', v1, v2

            x_name = 'time'
            y_name = 'canonical correlation'
            v1_name = self.names.values()[0][v1].split('_')[0]
            v2_name = self.names.values()[0][v2].split('_')[0]
            title = 'View-pairwise canonical correlation' + \
                ' (n time p frequency) over time for view ' + \
                v1_name + ' vs ' + v2_name

            for (i, (s, data)) in enumerate(dm.items()):

                print '\tGenerating plot for subject', s

                ax = fig.add_subplot(
                    len(self.subjects), 1, i+1)
                s_title = title + \
                    self.names[s][v1] + ' ' + self.names[s][v2]

                self._get_line_plot(s, v1, data, ax)

            plt.clf()

            fn = '_'.join(title.split()) + '.png'
            path = os.path.join(
                self.n_time_p_frequency_cc_dir, fn)

            fig.savefig(path, format='png')

    def _get_line_plot(self, s, v, data, ax):

        x_axis = None

        if self.clock_time:
            dl = self.servers[s][v].get_status()['data_loader']
            start_time = self.loaders[s][v].get_status()['start_times'][0]
            n = data.shape[0]
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
            x_axis = np.arange(data.shape[0])[:,np.newaxis]

        return ax.plot(x_axis, data)

    def _show_n_time_p_frequency(self):

        for (s, spud) in self.ccas[self.cca_names[0]].items():
            
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
                (v1_l, v2_l) = unzip(subperiods)
                title = 'View-pairwise cca (n time p frequency) for views ' + \
                    self.names[s][v1] + ' ' + self.names[s][v2] + \
                    ' of subject ' + s
                x_name = 'subperiod'
                y_name = 'dimension'
                v_name = 'canonical vector value'

                ax1 = fig.add_subplot(211)

                self._get_heat_plot(
                    s,
                    v1,
                    v1_l,
                    x_name,
                    y_name,
                    v_name,
                    ax1)

                ax2 = fig.add_subplot(212)

                self._get_heat_plot(
                    s,
                    v2,
                    v2_l,
                    x_name,
                    y_name,
                    v_name,
                    ax2)

                fig.suptitle(title)

                fn = '_'.join(title.split()) + '.png'
                path = os.path.join(
                    self.n_time_p_frequency_dir, fn)

                fig.savefig(path, format='png')

    def _get_heat_plot(self, s, v, ccal, x_name, y_name, v_name, ax):

        tl = np.hstack(ccal)

        yl = np.arange(self.p_by_view[v])
        xl = np.arange(tl.shape[1])

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
            n = tl.shape[0]
            dl = self.servers[s][v].get_status()['data_loader']
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
