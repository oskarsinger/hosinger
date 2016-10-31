import os
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat

class SubperiodCorrelationRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        cca=False,
        save=False,
        load=False,
        show=False,
        show_max=False):

        self.cca = cca
        self.save = save
        self.load = load
        self.show = show
        self.show_max = show_max

        self._init_dirs(
            save, 
            load, 
            show, 
            show_max,
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

        default = lambda: [[] for i in xrange(self.num_subperiods)]

        self.correlation = {s : SPUD(self.num_views, default=default)
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show_corr_over_periods()
            self._show_corr_over_subperiods()

        if self.show_max:
            self._show_corr_max_over_periods()
            self._show_corr_max_over_subperiods()

    def _init_dirs(self, 
        save, 
        load, 
        show, 
        show_max, 
        save_load_dir):

        if (show_max or show or save) and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            corr_or_cca = 'cca' if self.cca else 'corr'
            model_dir = get_ts('VPWSPCR' + corr_or_cca)

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.corr_dir = rmu.init_dir(
            'correlation',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show or show_max,
            self.save_load_dir) 

    def _compute(self):

        for (s, s_wavelets) in self.wavelets.items():
            spud = self.correlation[s]

            for (p, day) in enumerate(s_wavelets):
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) =  subperiod[k[0]]
                        (Yh2, Yl2) =  subperiod[k[1]]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                        correlation = None

                        if self.cca:
                            correlation = rmu.get_cca_vecs(
                                Y1_mat, Y2_mat)
                        else:
                            correlation = rmu.get_normed_correlation(
                                Y1_mat, Y2_mat)

                        self.correlation[s].get(k[0], k[1])[sp].append(
                            correlation)
                     
                        if self.save:
                            self._save(
                                correlation,
                                s,
                                k,
                                p,
                                sp)

    def _save(self, c, s, v, p, sp):

        views = str(v[0]) + '-' + str(v[1])
        path = '_'.join([
            'subject', s,
            'views', views,
            'period', str(p),
            'subperiod', str(sp)])
        path = os.path.join(self.corr_dir, path)

        with open(path, 'w') as f:
            np.save(f, c)

    def _load(self):

        correlation = {} 

        for fn in os.listdir(self.corr_dir):
            path = os.path.join(self.corr_dir, fn)

            with open(path) as f:
                correlation[fn] = np.load(f)

        for (s, spud) in self.correlation.items():
            for (k, subperiods) in spud.items():
                for i in xrange(self.num_subperiods):
                    subperiods[i] = [None] * self.num_periods[s] 
                
        for (k, m) in correlation.items():
            info = k.split('_')
            s = info[1]
            v = [int(i) for i in info[3].split('-')]
            p = int(info[5])
            sp = int(info[7])

            self.correlation[s].get(v[0], v[1])[sp][p] = m

    def _show_corr_max_over_subperiods(self):

        for (s, spud) in self.correlation.items():
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(self.num_views, default=default)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                (n, m) = periods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_periods[s])]
                timelines = [rmu.get_ravel_hstack(subperiods)
                             for subperiods in periods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])
                title = 'View-pairwise max-over-hours correlation' + \
                    ' over days for views ' + \
                    self.names[k[0]] + ' ' + self.names[k[1]] + \
                    ' of subject ' + s
                fn = '_'.join(title.split()) + '.pdf'
                path = os.path.join(self.plot_dir, fn)

                plot_matrix_heat(
                    timeline,
                    x_labels,
                    y_labels,
                    title,
                    'day',
                    'frequency pair',
                    'correlation',
                    vmax=1,
                    vmin=-1)[0].get_figure().savefig(
                        path, format='pdf')
                sns.plt.clf()

    def _show_corr_over_subperiods(self):

        for (s, spud) in self.correlation.items():
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(self.num_views, default=default)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                (n, m) = periods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_subperiods)]
                name1 = self.names[k[0]]
                name2 = self.names[k[1]]

                for (p, subperiods) in enumerate(periods):
                    timeline = rmu.get_ravel_hstack(subperiods)
                    title = 'View-pairwise correlation over hours ' + \
                        ' for views ' + name1 + ' ' + name2 + \
                        ' of subject ' + s + ' and day ' + \
                        rmu.get_2_digit(p)
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'hour',
                        'frequency pair',
                        'correlation',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()

    def _show_corr_max_over_periods(self):

        for (s, spud) in self.correlation.items():
            for (k, subperiods) in spud.items():
                (n, m) = subperiods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_subperiods)]
                timelines = [rmu.get_ravel_hstack(periods)
                             for periods in subperiods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])
                title = 'View-pairwise max-over-days correlation' + \
                    ' over hours for views ' + \
                    self.names[k[0]] + ' ' + self.names[k[1]] + \
                    ' of subject ' + s
                fn = '_'.join(title.split()) + '.pdf'
                path = os.path.join(self.plot_dir, fn)

                plot_matrix_heat(
                    timeline,
                    x_labels,
                    y_labels,
                    title,
                    'hour',
                    'frequency pair',
                    'correlation',
                    vmax=1,
                    vmin=-1)[0].get_figure().savefig(
                        path, format='pdf')
                sns.plt.clf()

    def _show_corr_over_periods(self):

        for (s, spud) in self.correlation.items():
            for (k, subperiods) in spud.items():
                (n, m) = subperiods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_periods[s])]

                for (sp, periods) in enumerate(subperiods):
                    timeline = rmu.get_ravel_hstack(periods)
                    title = 'View-pairwise correlation over days for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                        ' of subject ' + s + ' at hour ' + str(sp)
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'day',
                        'frequency pair',
                        'correlation',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()
