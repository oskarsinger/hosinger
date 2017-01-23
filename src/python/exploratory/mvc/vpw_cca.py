import os
import json
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.stats import get_cca_vecs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from lazyprojector import plot_matrix_heat, plot_lines
from math import log, ceil

class ViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        show=False,
        nnz=1):

        self.servers = servers
        self.show = show
        self.nnz = nnz

        self._init_dirs


        self.subjects = self.servers.keys()
        self.num_subperiods = num_subperiods
        self.subperiod = dtcwt_runner.subperiod
        self.names = [s.get_status()['data_loader'].name()
                      for s in self.servers.values()[0]]
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = len(self.servers.values()[0])
        self.num_periods = {s : int(servers[0].num_periods / self.num_subperiods)
                            for (s, servers) in self.servers.items()}
	self.max_periods = max(self.num_periods.values())
        self.cca_names = {
            'cca_over_time',
            'cca_over_freqs',
            'cc_over_time'}

        self.rates = dtcwt_runner.rates

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

            self._show_cca_over_periods()
            self._show_cca_over_subperiods()

            self._show_cca_mean_over_periods()
            self._show_cca_mean_over_subperiods()

            self._show_cca_over_freqs()

            self._show_cc()
        else:
            self._compute()

    def _init_dirs(self, 
        show, 
        save_load_dir):

        if show:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCCA')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir
            freqs_path = os.path.join(
                self.save_load_dir,
                'num_freqs.json')

            with open(freqs_path) as f:
                line = f.readline()

                self.num_freqs = json.loads(line)

        get_path = lambda n: os.path.join(
            self.save_load_dir, n)
        hdf5_paths = {n : get_path(n) for n in self.cca_names}
        self.hdf5_repos = {n : h5py.File(p, 'w' if save else 'r')
                           for p in hdf5_paths}
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir) 

    def _compute(self):

        for (s, servers) in self.servers.items():
            print 'Computing CCAs for subject', s

            for sp in enumerate(self.num_subperiods * self.num_periods[s]):
                subperiods = [s.get_data() for s in servers]

                for i in xrange(self.num_views):
                    for j in xrange(i, self.num_views):
                        v1_mat = subperiod[i]
                        v2_mat = subperiod[j]
                        cca_over_time = np.vstack(get_cca_vecs(
                            v1_mat, v2_mat))
                        cca_dim = min(v1_mat.shape + v2_mat.shape)
                        cca_over_freqs = np.hstack(get_cca_vecs(
                            v1_mat[:,:cca_dim].T,
                            v2_mat[:,:cca_dim].T,
                            num_nonzero=self.nnz))
                        cc_over_time = self._get_cc_over_time(
                            v1_mat,
                            v2_mat,
                            cca_over_time)
                        stuff = {
                            self.cca_names[0]: cca_over_time,
                            self.cca_names[1]: cca_over_freqs,
                            self.cca_names[2]: cc_over_time}

                        if p == 0:
                            self.num_freqs[i] = Y1_mat.shape[1] 
                            self.num_freqs[j] = Y2_mat.shape[1]

                        self._save(
                            stuff,
                            s,
                            i,
                            j,
                            sp)

        num_freqs_json = json.dumps(self.num_freqs)
        path = os.path.join(
            self.save_load_dir, 
            'num_freqs.json')

        with open(path, 'w') as f:
            f.write(num_freqs_json)

    def _get_cc_over_time(self, v1_mat, v2_mat, cca_over_time):

        v1_cc = np.dot(
            v1_mat, 
            cca_over_time[:v1_mat.shape[1],:])
        v2_cc = np.dot(
            v2_mat, 
            cca_over_time[v1_mat.shape[1]:,:])

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

    def _show_cca_over_freqs(self):

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
                unit = rmu.get_symptom_status(s) \
                    if self.subject_mean else None
                phi1 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[:,0][:,np.newaxis],
                    unit)
                phi2 = (
                    factor * np.arange(tl.shape[0])[:,np.newaxis], 
                    tl[:,1][:,np.newaxis],
                    unit)
                data_maps.get(k[0], k[1])[s_key + str(1)] = phi1
                data_maps.get(k[0], k[1])[s_key + str(2)] = phi2

        for (k, dm) in data_maps.items():
            x_name = 'time (days)'
            y_name = 'canonical vector value'
            title = 'View-pairwise canonical vector values' + \
                ' over frequencies for views ' + \
                self.names[k[0]] + ' ' + self.names[k[1]]
            unit_name = 'Symptomatic?' \
                if self.subject_mean else None

            self._line_plot_save_clear(
                dm,
                x_name,
                y_name,
                title,
                unit_name=unit_name)

    def _show_cc(self):

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

        for (k, dm) in data_maps.items():
            x_name = 'time'
            y_name = 'canonical correlation'
            title = 'View-pairwise canonical correlation' + \
                ' over time for views ' + \
                self.names[k[0]] + ' ' + self.names[k[1]]

            self._line_plot_save_clear(
                dm,
                x_name,
                y_name,
                title)

    def _get_tl_spuds(self, index):

        tl_spuds = {s: SPUD(self.num_views, no_double=True)
                    for s in self.ccas.keys()}

        for (s, spud) in self.ccas.items():
            cc_over_time = SPUD(
                self.num_views, 
                default=lambda: [None] * self.num_periods[s],
                no_double=True)

            for (k, subperiods) in spud.items():
                num_freqs = min([
                    self.num_freqs[k[0]],
                    self.num_freqs[k[1]]])
                rate = max([
                    self.rates[k[0]],
                    self.rates[k[1]]])
                full_length = int(
                    rate * self.subperiod / 2**(num_freqs)) # - 1))

                for (sp, periods) in enumerate(subperiods):
                    for (p, period) in enumerate(periods):
                        tls = cc_over_time.get(k[0], k[1])
                        p_over_time = period[index]

                        if p_over_time.shape[0] < full_length:
                            padding_l = full_length - p_over_time.shape[0]

                            if index == 1:
                                padding = np.array(
                                    [[np.nan,np.nan]] * padding_l)
                            else:
                                padding = np.array(
                                    [np.nan] * padding_l)[:,np.newaxis]

                            p_over_time = np.vstack(
                                [p_over_time, padding])

                        if tls[p] is None:
                            tls[p] = p_over_time
                        else:
                            tls[p] = np.vstack(
                                [tls[p], p_over_time])

            for (k, tls) in cc_over_time.items():
                tl = np.vstack(tls)
                cc_over_time.insert(k[0], k[1], tl)

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

    def _show_cca_mean_over_subperiods(self):

        means = None
        counts = None

        if self.subject_mean:
            keys = {rmu.get_symptom_status(s)
                    for s in self.subjects}
            means = {k: SPUD(self.num_views, no_double=True)
                     for k in keys}
            counts = {k: SPUD(
                        self.num_views, 
                        default=lambda:0, 
                        no_double=True)
                      for k in keys}
            p = max(self.num_periods.values())

            for spud in means.values():
                for (k1, k2) in spud.keys():
                    n = self.num_freqs[k1] + self.num_freqs[k2]

                    spud.insert(k1, k2, np.zeros((n, p)))

        for (s, spud) in self.ccas.items():
            status = rmu.get_symptom_status(s)
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_ccas = SPUD(
                self.num_views, 
                default=default,
                no_double=True)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, cca) in enumerate(periods):
                        period_ccas.get(k[0], k[1])[p].append(cca)

            for (k, periods) in period_ccas.items():
                timelines = [np.hstack([sp[0] for sp in subperiods])
                             for subperiods in periods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])

                if self.subject_mean:
                    tl_shape = timeline.shape
                    avg_shape = means[status].get(k[0], k[1]).shape

                    if tl_shape == avg_shape:
                        count = counts[status].get(k[0], k[1]) + 1
                        counts[status].insert(
                            k[0], k[1], count)

                        avg = get_ra(
                            means[status].get(k[0], k[1]),
                            timeline,
                            count)

                        means[status].insert(
                            k[0], k[1], avg)
                else:
                    (y_labels, x_labels) = self._get_labels(
                        k[0], k[1], self.num_periods[s])
                    title = 'View-pairwise mean-over-hours cca' + \
                        ' over days for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                        ' of subject ' + s

                    self._matrix_plot_save_clear(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'day')

        for (status, spud) in means.items():
            for (k, avg) in spud.items():
                (y_labels, x_labels) = self._get_labels(
                    k[0], k[1], self.num_periods[s])
                title = status + ' view-pairwise mean-over-hours' + \
                    ' cca over days for views ' + \
                    self.names[k[0]] + ' ' + self.names[k[1]]

                self._matrix_plot_save_clear(
                    avg,
                    x_labels,
                    y_labels,
                    title,
                    'day')

    def _show_cca_over_subperiods(self):

        for (s, spud) in self.correlation.items():
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(
                self.num_views, 
                default=default,
                no_double=True)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                (y_labels, x_labels) = self._get_labels(
                    k[0], k[1], self.num_subperiods)
                name1 = self.names[k[0]]
                name2 = self.names[k[1]]

                for (p, subperiods) in enumerate(periods):
                    timeline = np.hstack([sp[0] for sp in subperiods])
                    title = 'View-pairwise cca over hours ' + \
                        ' for views ' + name1 + ' ' + name2 + \
                        ' of subject ' + s + ' and day ' + \
                        rmu.get_2_digit(p)

                    self._matrix_plot_save_clear(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'hour')

    def _show_cca_mean_over_periods(self):

        means = None
        counts = None

        if self.subject_mean:
            keys = {rmu.get_symptom_status(s)
                    for s in self.subjects}
            means = {k: SPUD(self.num_views, no_double=True)
                     for k in keys}
            counts = {k: SPUD(
                        self.num_views, 
                        default=lambda:0, 
                        no_double=True)
                      for k in keys}
            p = self.num_subperiods

            for spud in means.values():
                for (k1, k2) in spud.keys():
                    n = self.num_freqs[k1] + self.num_freqs[k2]

                    spud.insert(k1, k2, np.zeros((n, p)))

        for (s, spud) in self.ccas.items():
            status = rmu.get_symptom_status(s)

            for (k, subperiods) in spud.items():
                timelines = [np.hstack([p[0] for p in periods])
                             for periods in subperiods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])

                if self.subject_mean:
                    tl_shape = timeline.shape
                    avg_shape = means[status].get(k[0], k[1]).shape

                    if tl_shape == avg_shape:
                        count = counts[status].get(k[0], k[1]) + 1
                        counts[status].insert(
                            k[0], k[1], count)

                        avg = get_ra(
                            means[status].get(k[0], k[1]),
                            timeline,
                            count)

                        means[status].insert(
                            k[0], k[1], avg)
                else:
                    (y_labels, x_labels) = self._get_labels(
                        k[0], k[1], self.num_subperiods)
                    title = 'View-pairwise mean-over-days cca' + \
                        ' over hours for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                        ' of subject ' + s

                    self._matrix_plot_save_clear(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'hour')

        for (status, spud) in means.items():
            for (k, avg) in spud.items():
                (y_labels, x_labels) = self._get_labels(
                    k[0], k[1], self.num_subperiods)
                title = status + ' view-pairwise mean-over-days' + \
                    ' cca over hours for views ' + \
                    self.names[k[0]] + ' ' + self.names[k[1]]

                self._matrix_plot_save_clear(
                    avg,
                    x_labels,
                    y_labels,
                    title,
                    'hour')

    def _show_cca_over_periods(self):

        for (s, spud) in self.ccas.items():
            for (k, subperiods) in spud.items():
                (y_labels, x_labels) = self._get_labels(
                    k[0], k[1], self.num_periods[s])

                for (sp, periods) in enumerate(subperiods):
                    timeline = np.hstack([p[0] for p in periods])
                    title = 'View-pairwise cca over days for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                        ' of subject ' + s + ' at hour ' + str(sp)

                    self._matrix_plot_save_clear(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'day')

    def _matrix_plot_save_clear(self, 
        timeline, 
        x_labels, 
        y_labels, 
        title,
        day_or_hour):

        fn = '_'.join(title.split()) + '.pdf'
        path = os.path.join(self.plot_dir, fn)

        plot_matrix_heat(
            timeline,
            x_labels,
            y_labels,
            title,
            day_or_hour,
            'frequency component canonical basis value and view',
            'cca',
            vmax=1,
            vmin=-1)[0].get_figure().savefig(
                path, format='pdf')
        sns.plt.clf()

    def _get_labels(self, view1, view2, x_len):

        n1 = self.num_freqs[view1]
        n2 = self.num_freqs[view2]
        y1_labels = ['view ' + str(view1) + ' ' + rmu.get_2_digit(i)
                     for i in xrange(n1)]
        y2_labels = ['view ' + str(view2) + ' ' + rmu.get_2_digit(i)
                     for i in xrange(n2)]
        y_labels = y1_labels + y2_labels
        x_labels = [rmu.get_2_digit(p, power=False)
                    for p in xrange(x_len)]

        return (y_labels, x_labels)
