import os
import json
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from drrobert.arithmetic import get_running_avg as get_ra
from lazyprojector import plot_matrix_heat
from math import log

class ViewPairwiseCCARunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        save=False,
        load=False,
        show=False,
        show_mean=False,
        subject_mean=False,
        show_transpose=False,
        show_cc=False):

        self.save = save
        self.load = load
        self.show = show
        self.show_mean = show_mean
        self.subject_mean = subject_mean
        self.show_transpose = show_transpose
        self.show_cc = show_cc

        self.wavelets = dtcwt_runner.wavelets
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps
        self.num_freqs = [None] * self.num_views

        self._init_dirs(
            save, 
            load, 
            show, 
            show_mean,
            save_load_dir)

        default = lambda: [[] for i in xrange(self.num_subperiods)]

        self.ccas = {s : SPUD(
                        self.num_views, 
                        default=default, 
                        no_double=True)
                    for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show_cca_over_periods()
            self._show_cca_over_subperiods()

        if self.show_mean:
            self._show_cca_mean_over_periods()
            self._show_cca_mean_over_subperiods()

        if self.show_cc:
            self._show_cc()

    def _init_dirs(self, 
        save, 
        load, 
        show, 
        show_mean, 
        save_load_dir):

        if (show_mean or show or save) and not load:
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

        self.ccas_dir = rmu.init_dir(
            'cca',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show or show_mean,
            self.save_load_dir) 

    def _compute(self):

        for (s, s_wavelets) in self.wavelets.items():
            spud = self.ccas[s]

            for (p, day) in enumerate(s_wavelets):
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) = subperiod[k[0]]
                        (Yh2, Yl2) = subperiod[k[1]]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                        cca_over_time = rmu.get_cca_vecs(
                            Y1_mat, Y2_mat)
                        cca_dim = min([Y1_mat.shape[1], Y2_mat.shape[1]])
                        cca_over_freqs = rmu.get_cca_vecs(
                            Y1_mat[:,cca_dim].T,
                            Y2_mat[:,cca_dim].T,
                            num_nonzero=cca_dim)
                        cc_over_time = np.dot(
                            np.hstack([Y1_mat, Y2_mat]), 
                            cca_over_time)
                        stuff = (
                            cca_over_time,
                            cca_over_freqs,
                            cc_over_time)

                        if p == 0:
                            self.num_freqs[k[0]] = Y1_mat.shape[1] 
                            self.num_freqs[k[1]] = Y2_mat.shape[1]

                        self.ccas[s].get(k[0], k[1])[sp].append(stuff)
                     
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

    def _save(self, cs, s, v, p, sp):

        views = str(v[0]) + '-' + str(v[1])
        path = '_'.join([
            'subject', s,
            'views', views,
            'period', str(p),
            'subperiod', str(sp)])
        path = os.path.join(self.ccas_dir, path)

        with open(path, 'w') as f:
            np.savez(f, *cs)

    def _load(self):

        cca = {} 

        for fn in os.listdir(self.ccas_dir):
            path = os.path.join(self.ccas_dir, fn)

            with open(path) as f:
                cca[fn] = np.load(f)

        for (s, spud) in self.ccas.items():
            for (k, subperiods) in spud.items():
                for i in xrange(self.num_subperiods):
                    subperiods[i] = [None] * self.num_periods[s] 
                
        for (k, l) in cca.items():
            info = k.split('_')
            s = info[1]
            v = [int(i) for i in info[3].split('-')]
            p = int(info[5])
            sp = int(info[7])
            loaded = {int(h_fn.split('_')[1]) : a
                      for (h_fn, a) in loaded.items()}
            num_coeffs = len(loaded)
            coeffs = [loaded[i] 
                      for i in xrange(num_coeffs)]

            self.ccas[s].get(v[0], v[1])[sp][p] = tuple(coeffs)

    def _show_cc(self):

        for (s, spud) in self.ccas.items():
            cc_over_time = SPUD(
                self.num_views, 
                default=lambda: [None] * self.num_periods[s],
                no_double=True)
            for (k, subperiods) in spud.items():
                for (sp, periods) in enumerate(subperiods):
                    for (p, period) in enumerate(periods):
                        tl = cc_over_time.get(k[0], k[1])

                        if tl[p] is None:
                            tl[p] = period
                        else:
                            tl[p] = np.vstack([tl[p], period])

        # TODO: do the actual plotting here

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

                    self._plot_save_clear(
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

                self._plot_save_clear(
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

                    self._plot_save_clear(
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

                    self._plot_save_clear(
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

                self._plot_save_clear(
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

                    self._plot_save_clear(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'day')

    def _plot_save_clear(self, 
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
