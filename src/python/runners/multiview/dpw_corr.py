import os
import seaborn as sns

import numpy as np
import utils as rmu

from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat

class DayPairwiseCorrelationRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        cca=False,
        save=False,
        load=False,
        show=False):

        self.cca = cca
        self.save = save
        self.load = load
        self.show = show

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.correlation = {s : [[[] for i in xrange(self.num_subperiods)]
                                 for i in xrange(self.num_views)]
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show_corr_over_periods()
            self._show_corr_over_subperiods()

    def _init_dirs(self, save, load, show, save_load_dir):

        if (save or show) and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            corr_or_cca = 'cca' if self.cca else 'corr'
            model_dir = get_ts('DPWCR' + corr_or_cca)

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
            show,
            self.save_load_dir) 

    def _compute(self):

        for subject in self.subjects:

            s_wavelets = self.wavelets[subject]
            day_pairs = zip(
                s_wavelets[:-1],
                s_wavelets[1:])

            for (p, (day1, day2)) in enumerate(day_pairs):
                iterable = enumerate(zip(day1, day2))

                for (sp, (sp1, sp2)) in iterable:
                    for v in xrange(self.num_views):
                        (Yh1, Yl1) = sp1[v]
                        (Yh2, Yl2) = sp2[v]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                        correlation = None

                        if self.cca:
                            correlation = rmu.get_cca_vecs 
                        else:
                            correlation = rmu.get_normed_correlation(
                                Y1_mat, Y2_mat)

                        self.correlation[subject][v][sp].append(
                            correlation)

                        if self.save:
                            self._save(
                                correlation,
                                subject,
                                v,
                                p,
                                sp)

    def _save(self, c, s, v, p, sp):
        path = '_'.join([
            'subject', s,
            'view', self.names[v],
            'periods', str(p) + '-' + str(p+1),
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

        for (s, views) in self.correlation.items():
            for i in xrange(len(views)):
                l = [[None] * self.num_periods[s]
                     for i in xrange(self.num_subperiods)]

                self.correlation[s][i] = l

        for (k, m) in correlation.items():
            info = k.split('_')
            s = info[1]
            v = self.names2indices[info[3]]
            ps = [int(i) for i in info[5].split('-')]
            sp = int(info[7])

            self.correlation[s][v][sp][ps[0]] = m

    def _save_corr_over_subperiods(self):

        for (s, views) in self.correlation.items():
            period_corrs = [[[] for p in xrange(self.num_periods[s])] 
                            for v in xrange(self.num_views)]

            for (v, subperiods) in enumerate(views):
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs[v][p].append(corr)

            for (v, periods) in enumerate(periods_corrs):
                (n, m) = periods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(str(sp))
                            for sp in xrange(self.num_subperiods)]

                for (p, subperiods) in enumerate(periods):
                    timeline = rmu.get_ravel_hstack(subperiods)
                    title = 'Day-pairwise correlation over hours ' + \
                        ' for view ' + self.names[v] + \
                        ' of subject ' + s + ' and day pair ' + \
                        rmu.get_2_digit_pair(p, p+1)
                    fn = '_'.join(title.split()) + '.png'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'hour',
                        'frequency pair',
                        'correlation')[0].get_figure().savefig(path)
                    sns.plt.clf()
        
    def _save_corr_over_periods(self):

        for (s, views) in self.correlation.items():
            for (v, subperiods) in enumerate(views):
                (n, m) = subperiods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit_pair(p, p+1)
                            for p in xrange(self.num_periods[s]-1)]

                for (sp, periods) in enumerate(subperiods):
                    timeline = rmu.get_ravel_hstack(periods)
                    title = 'Day-pairwise correlation over day pairs ' + \
                        ' for view ' + self.names[v] + \
                        ' of subject ' + s + ' at subperiod ' + str(sp)
                    fn = '_'.join(title.split()) + '.png'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'period pair',
                        'frequency pair',
                        'correlation')[0].get_figure().savefig(path)
                    sns.plt.clf()
