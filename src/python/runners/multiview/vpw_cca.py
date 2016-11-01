import os
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat

class ViewPairwiseCCARunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        save=False,
        load=False,
        show=False,
        show_mean=False):

        self.save = save
        self.load = load
        self.show = show
        self.show_mean = show_mean

        self._init_dirs(
            save, 
            load, 
            show, 
            show_mean,
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        
        print len(self.wavelets.values())
        print len(self.wavelets.values()[0])
        print len(self.wavelets.values()[0][0])
        sample = self.wavelets.values()[0][0][0]

        self.num_freqs = [len(sample[i][0]) + 1
                          for i in xrange(self.num_views)]
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

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
                        (Yh1, Yl1) =  subperiod[k[0]]
                        (Yh2, Yl2) =  subperiod[k[1]]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                        cca = rmu.get_cca_vecs(Y1_mat, Y2_mat)

                        self.ccas[s].get(k[0], k[1])[sp].append(cca)
                     
                        if self.save:
                            self._save(
                                cca,
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
        path = os.path.join(self.ccas_dir, path)

        with open(path, 'w') as f:
            np.save(f, c)

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
                
        for (k, m) in cca.items():
            info = k.split('_')
            s = info[1]
            v = [int(i) for i in info[3].split('-')]
            p = int(info[5])
            sp = int(info[7])

            self.ccas[s].get(v[0], v[1])[sp][p] = m

    def _show_cca_mean_over_subperiods(self):

        for (s, spud) in self.ccas.items():
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
                n1 = self.num_freqs[k[0]]
                n2 = self.num_freqs[k[1]]
                y1_labels = [rmu.get_2_digit(i) + ' view ' + str(k[0])
                             for i in xrange(n1)]
                y2_labels = [rmu.get_2_digit(i) + ' view ' + str(k[1])
                             for i in xrange(n2)]
                y_labels = y1_labels + y2_labels
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_periods[s])]
                timelines = [np.hstack(subperiods)
                             for subperiods in periods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])
                title = 'View-pairwise mean-over-hours cca' + \
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
                    'frequency component and view',
                    'cca parameters',
                    vmax=1,
                    vmin=-1)[0].get_figure().savefig(
                        path, format='pdf')
                sns.plt.clf()

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
                n1 = self.num_freqs[k[0]]
                n2 = self.num_freqs[k[1]]
                y1_labels = [rmu.get_2_digit(i) + ' view ' + str(k[0])
                             for i in xrange(n1)]
                y2_labels = [rmu.get_2_digit(i) + ' view ' + str(k[1])
                             for i in xrange(n2)]
                y_labels = y1_labels + y2_labels
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_subperiods)]
                name1 = self.names[k[0]]
                name2 = self.names[k[1]]

                for (p, subperiods) in enumerate(periods):
                    timeline = np.hstack(subperiods)
                    title = 'View-pairwise cca over hours ' + \
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
                        'frequency component and view',
                        'cca',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()

    def _show_cca_mean_over_periods(self):

        for (s, spud) in self.ccas.items():
            for (k, subperiods) in spud.items():
                n1 = self.num_freqs[k[0]]
                n2 = self.num_freqs[k[1]]
                y1_labels = [rmu.get_2_digit(i) + ' view ' + str(k[0])
                             for i in xrange(n1)]
                y2_labels = [rmu.get_2_digit(i) + ' view ' + str(k[1])
                             for i in xrange(n2)]
                y_labels = y1_labels + y2_labels
                print 'y_labels', y_labels
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_periods[s])]
                print 'x_labels', x_labels
                timelines = [np.hstack(periods)
                             for periods in subperiods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])
                title = 'View-pairwise mean-over-days cca' + \
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
                    'frequency component and view',
                    'cca',
                    vmax=1,
                    vmin=-1)[0].get_figure().savefig(
                        path, format='pdf')
                sns.plt.clf()

    def _show_cca_over_periods(self):

        for (s, spud) in self.ccas.items():
            for (k, subperiods) in spud.items():
                n1 = len(self.wavelets.values()[0][0][0][k[0]][0]) + 1
                n2 = len(self.wavelets.values()[0][0][0][k[1]][0]) + 1
                y1_labels = [rmu.get_2_digit(i) + ' view ' + str(k[0])
                             for i in xrange(n1)]
                y2_labels = [rmu.get_2_digit(i) + ' view ' + str(k[1])
                             for i in xrange(n2)]
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_periods[s])]

                for (sp, periods) in enumerate(subperiods):
                    timeline = np.hstack(periods)
                    title = 'View-pairwise cca over days for views ' + \
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
                        'frequency component and view',
                        'cca',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()
