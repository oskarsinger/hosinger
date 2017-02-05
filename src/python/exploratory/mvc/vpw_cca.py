import os
import json
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.stats import get_cca_vecs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from drrobert.misc import unzip
from lazyprojector import plot_matrix_heat, plot_lines
from math import log, ceil

class ViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        nnz=1,
        show=False):

        self.servers = servers
        self.show = show
        self.nnz = nnz

        self._init_dirs


        self.subjects = self.servers.keys()
        self.num_subperiods = num_subperiods
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

                    for j in xrange(i, self.num_views):
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

                sp_group = v_group[sp]
                (Phi1, Phi2) = cs[n]

                sp_group.create_dataset('1', Phi1)
                sp_group.create_dataset('2', Phi2)
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
                    vs = [int(v) for v in v_str.split('-')]
                    cca_vs = cca_s.get(vs[0], vs[1])

                    for (sp_str, sp_group) in v_group.items():
                        sp = int(sp_str)

                        if n in self.cca_names[:2]:
                            cca_vs[sp] = (
                                np.array(sp_group['1']),
                                np.array(sp_group['2']))
                        else:
                            cca_vs[sp] = np.array(sp_group)

    def _show_n_frequency_p_time(self):

        tl_spuds = self._get_tl_spuds(1)
        default = lambda: {}
        data_maps = SPUD(
            self.num_views,
            default=default,
            no_double=True)

        for (s, spud) in tl_spuds.items():
            for (k, tl) in spud.items():
                s_key = 'Subject ' + s + ' view '
                factor = float(self.num_periods[s]) / tl.shape[0]
                phi1 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[:,0][:,np.newaxis],
                    None)
                phi2 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[:,1][:,np.newaxis],
                    None)
                data_maps.get(k[0], k[1])[s_key + str(1)] = phi1
                data_maps.get(k[0], k[1])[s_key + str(2)] = phi2

        fig = plt.figure()
        
        for ((v1, v2), dm) in data_maps.items():
            x_name = 'time (days)'
            y_name = 'canonical vector value'
            title = 'View-pairwise canonical vectors' + \
                ' (n frequency p time) for views '

            for (i, (s, data)) in enumerate(dm.items()):
                ax = fig.add_subplot(
                    len(self.subjects), 1, i+1)
                s_title = title + \
                    self.names[s][v1] + ' ' + self.names[s][v2]
                s_dm = {s : data}

                self._line_plot_save_clear(
                    s_dm,
                    x_name,
                    y_name,
                    s_title)

            plt.clf()

    def _show_n_time_p_frequency_cc(self):

        tl_spuds = self._get_tl_spuds(2)
        default = lambda: {'Subject ' + s: None for s in self.subjects}
        data_maps = SPUD(
            self.num_views,
            default=default,
            no_double=True)

        for (s, spud) in tl_spuds.items():
            for (k, tl) in spud.items():
                s_key = 'Subject ' + s
                data = (
                    np.arange(len(tl))[:,np.newaxis], 
                    np.array(tl),
                    None)
                data_maps.get(k[0], k[1])[s_key] = data

        fig = plt.figure()

        for ((v1, v2), dm) in data_maps.items():
            x_name = 'time'
            y_name = 'canonical correlation'
            title = 'View-pairwise canonical correlation' + \
                ' (n time p frequency) over time for views '

            for (i, (s, data)) in enumerate(dm.items()):
                ax = fig.add_subplot(
                    len(self.subjects), 1, i+1)
                s_title = title + \
                    self.names[s][v1] + ' ' + self.names[s][v2]
                s_dm = {s : data}

                plot_lines(
                    s_dm, 
                    x_name, 
                    y_name, 
                    s_title)

            plt.clf()

    def _get_tl_spuds(self, index):

        tl_spuds = {s: SPUD(self.num_views, no_double=True)
                    for s in self.ccas.keys()}

        for (s, spud) in self.ccas[self.cca_names[index]].items():
            n_time_p_frequency_cc = SPUD(
                self.num_views, 
                default=lambda: [None] * self.num_periods[s],
                no_double=True)

            for ((v1, v2), subperiods) in spud.items():
                for (sp, periods) in enumerate(subperiods):
                    for (p, period) in enumerate(periods):
                        tls = n_time_p_frequency_cc.get(v1, v2)
                        p_over_time = period[index]

                        if tls[p] is None:
                            tls[p] = p_over_time
                        else:
                            tls[p] = np.vstack(
                                [tls[p], p_over_time])

            for (k, tls) in cc_over_time.items():
                tl = np.vstack(tls)
                n_time_p_frequency_cc.insert(v1, v2, tl)

            tl_spuds[s] = cc_over_time

        return tl_spuds

    def _line_plot_save_clear(self,
        dm,
        x_name,
        y_name,
        title,
        unit_name=None):

        fn = '_'.join(title.split()) + '.pdf'
        path = os.path.join(self.plot_dir, fn)

        plot_lines(
            dm, 
            x_name, 
            y_name, 
            title,
            unit_name=unit_name).get_figure().savefig(
            path, format='pdf')
        sns.plt.clf()

    def _show_n_time_p_frequency(self):

        for (s, spud) in self.ccas[self.cca_names[0]].items():
            for ((v1, v2), subperiods) in spud.items():
                fig = plt.figure()
                filename = '_'.join([
                    'subject', s,
                    '_views_',
                    self.names[s][v1] + '-' + self.names[s][v2]]) + '.png'
                path = os.path.join(
                    self.n_time_p_frequency_dir, filename)
                (v1_l, v2_l) = unzip(
                    [(pair['1'], pair['2']) for pair in subperiods])
                v1_tl = np.hstack(v1_l)
                v2_tl = np.hstack(v2_l)
                title = 'View-pairwise cca (n time p frequency) for views ' + \
                    self.names[s][v1] + ' ' + self.names[s][v2] + \
                    ' of subject ' + s + ' at subperiod ' + str(sp)
                x_name = 'subperiod'
                y_name = 'dimension'
                v_name = 'canonical vector value'
                ((yl1, yl2), xl) = self._get_labels(
                    v1, v2, self.num_periods[s])

                ax1 = fig.add_subplot(211)

                plot_matrix_heat(
                    tl1,
                    xl,
                    yl1,
                    '',
                    x_name,
                    y_name,
                    v_name,
                    vmax=1,
                    vmin=-1,
                    ax=ax1)

                ax2 = fig.add_subplot(212)

                plot_matrix_heat(
                    tl2,
                    xl,
                    yl2,
                    '',
                    x_name,
                    y_name,
                    v_name,
                    vmax=1,
                    vmin=-1,
                    ax=ax2)

                fn = '_'.join(title.split()) + '.png'
                path = os.path.join(
                    self.n_time_p_frequency_dir, fn)

                fig.savefig(path, format='png')

    def _get_labels(self, view1, view2, x_len):

        n1 = self.p_by_view[view1]
        n2 = self.p_by_view[view2]
        y1_labels = ['view ' + str(view1) + ' {:02d}'.format(i)
                     for i in xrange(n1)]
        y2_labels = ['view ' + str(view2) + ' {:02d}'.format(i)
                     for i in xrange(n2)]
        y_labels = (y1_labels, y2_labels)
        x_labels = ['{:02d}'.format(p)
                    for p in xrange(x_len)]

        return (y_labels, x_labels)
