import os
import json
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from drrobert.arithmetic import get_running_avg as get_ra
from lazyprojector import plot_matrix_heat, plot_lines
from math import log, ceil

class ViewPairwiseCCARunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        save=False,
        load=False,
        show=False,
        subject_mean=False,
        nnz=1):

        self.save = save
        self.load = load
        self.show = show
        self.subject_mean = subject_mean
        self.nnz = nnz

        self.wavelets = dtcwt_runner.wavelets
        self.subperiod = dtcwt_runner.subperiod
        self.subjects = dtcwt_runner.subjects
        self.rates = dtcwt_runner.rates
        self.names = dtcwt_runner.names
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps
        self.num_freqs = [None] * self.num_views
        self.cca_names = {
            'cca_over_time',
            'cca_over_freqs',
            'cc_over_time'}

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        default = lambda: [[] for i in xrange(self.num_subperiods)]
        get_container = lambda: {s : SPUD(
                                    self.num_views, 
                                    default=default, 
                                    no_double=True)
                                 for s in self.subjects}
        self.ccas = {n : get_container()
                     for n in self.cca_names}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show_cca_over_periods()
            #self._show_cca_over_subperiods()

            self._show_cca_over_freqs()
            self._show_cc()

    def _init_dirs(self, 
        save, 
        load, 
        show, 
        save_load_dir):

        mk_sl_dir = show or save

        if mk_sl_dir and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCCAR')

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

        for (s, s_wavelets) in self.wavelets.items():
            spud = self.ccas[s]

            for (p, day) in enumerate(s_wavelets):
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) = subperiod[k[0]]
                        (Yh2, Yl2) = subperiod[k[1]]
                        Y1_mat = rmu.get_padded_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_padded_wavelets(Yh2, Yl2)
                        cca_over_time = rmu.get_cca_vecs(
                            Y1_mat, Y2_mat)
                        cca_dim = min(Y1_mat.shape + Y2_mat.shape)
                        cca_over_freqs = rmu.get_cca_vecs(
                            Y1_mat[:,:cca_dim].T,
                            Y2_mat[:,:cca_dim].T,
                            num_nonzero=self.nnz)
                        cc_over_time = self._get_cc_over_time(
                            Y1_mat,
                            Y2_mat,
                            cca_over_time)
                        stuff = {
                            self.cca_names[0]: cca_over_time,
                            self.cca_names[1]: cca_over_freqs,
                            self.cca_names[2]: cc_over_time}

                        if p == 0:
                            self.num_freqs[k[0]] = Y1_mat.shape[1] 
                            self.num_freqs[k[1]] = Y2_mat.shape[1]

                        #self.ccas[s].get(k[0], k[1])[sp].append(stuff)
                     
                        if self.save:
                            self._save(
                                stuff,
                                s,
                                k,
                                p,
                                sp)

        if self.save:
            num_freqs_json = json.dumps(self.num_freqs)
            path = os.path.join(
                self.save_load_dir, 
                'num_freqs.json')

            with open(path, 'w') as f:
                f.write(num_freqs_json)

    def _get_cc_over_time(self, Y1_mat, Y2_mat, cca_over_time):

        Y1_cc = np.dot(
            Y1_mat, 
            cca_over_time[:Y1_mat.shape[1],:])
        Y2_cc = np.dot(
            Y2_mat, 
            cca_over_time[Y1_mat.shape[1]:,:])

        return Y1_cc * Y2_cc

    def _save(self, cs, s, v, p, sp):

        for (n, repo) in self.hdf5_repos.items():
            if s not in repo:
                repo.create_group(s)

            s_group = repo[s]
            v_str = str(v)
            
            if v_str not in s_group:
                s_group.create_group(v_str)

            v_group = s_group[v_str]
            sp_str = str(sp)

            if sp_str not in v_group:
                v_group.create_group(sp_str)

            sp_group = v_group[sp]
            p_str = str(p)

            if n in self.cca_names[:2]:
                (Phi1, Phi2) = cs[n]

                sp_group.create_group(p_str)
                p_group = sp_group[p_str]
                p_group.create_dataset('1', Phi1)
                p_group.create_dataset('2', Phi2)
            else:
                sp_group.create_dataset(p_str, data=cs[n])


    def _load(self):

        for (n, n_repo) in self.hdf5_repos.items():
            cca = self.ccas[n]
            for (s, spud) in cca.items():
                for (k, subperiods) in spud.items():
                    for i in xrange(self.num_subperiods):
                        subperiods[i] = [None] * self.num_periods[s] 
            
            for (s, s_group) in n_repo.items():
                cca_s = cca[s]

                for (v_str, v_group) in s_group.items():
                    vs = [int(v) for v in v_str.split('-')]
                    cca_vs = cca_s.get(vs[0], vs[1])

                    for (sp_str, sp_group) in v_group.items():
                        sp = int(sp_str)
                        cca_sp = cca_vs[sp]

                        for (p_str, p_group) in sp_group.items():
                            p = str(p_str)
                            data = None

                            if n in self.cca_names[:2]:
                                cca_sp[p] = (
                                    np.array(p_group['1']),
                                    np.array(p_group['2']))
                            else:
                                cca_sp[p] = np.array(p_group)

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
