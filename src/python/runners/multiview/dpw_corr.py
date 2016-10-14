import os
import seaborn

import numpy as np
import pandas as pd
import utils as rmu
import matplotlib.pyplot as plt

from drrobert.file_io import get_timestamped as get_ts

class DayPairwiseCorrelationRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        save=False,
        load=False,
        show=False):

        self.wavelets = dtcwt_runner.wavelets
        self.save = save
        self.load = load
        self.show = show

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.correlation = {s : [[] for i in xrange(self.num_views)]
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show()

    def _init_dirs(self, save, load, show, save_load_dir):

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('DPWCR')

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
                        (Yh1, Yl1) =  sp1[view]
                        (Yh2, Yl2) =  sp2[view]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                        correlation = rmu.get_normed_correlation(
                            Y1_mat, Y2_mat)

                        self.correlation[subject][view].append(
                            correlation)

                        if self.save:
                            self._save(
                                correlation,
                                subject,
                                view,
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
                l = [None] * (self.num_periods[s] - 1)

                self.correlation[s][i] = l

        for (k, m) in correlation.items():
            info = k.split('_')
            s = info[1]
            v = self.names2indices[info[3]]
            ps = [int(i) for i in info[5].split('-')]
            sp = int(info[7])

            self.correlation[s][v][ps[0]][sp] = m
        
    def _show(self):

        for (s, views) in self.correlation.items():
            for (view, periods) in enumerate(views):
                freq_pairs = []
                period_pairs = []
                correlation = []
                
                for (p, corr) in enumerate(periods):
                    (n, m) = corr.shape
                    period_pair = str(p) + ', ' + str(p+1)

                    for i in xrange(n):
                        exp = str(i)
                        exp = '0' + exp if len(exp) == 1 else exp
                        freq_i = '2^' + exp

                        for j in xrange(m):
                            correlation.append(corr[i,j])
                            period_pairs.append(period_pair)
                            
                            exp = str(j)
                            exp = '0' + exp if len(exp) == 1 else exp
                            freq_j = '2^' + exp

                            freq_pairs.append(
                                freq_i + ', ' + freq_j)
                d = {
                    'freq_pairs': freq_pairs,
                    'period_pairs': period_pairs,
                    'correlation': correlation}
                df = pd.DataFrame(data=d)
                df = df.pivot(
                    'freq_pairs',
                    'period_pairs',
                    'correlation')
                ax = plt.axes()
                plot = seaborn.heatmap(
                    df,
                    yticklabels=8,
                    ax=ax)
                ax.set_title(
                    'Day-pair autocorrelation of view ' + 
                    self.names[view] + 
                    ' for subject ' + s)

                for label in plot.get_yticklabels():
                    label.set_rotation(45)

                seaborn.plt.show()   
