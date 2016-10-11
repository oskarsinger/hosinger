import os
import seaborn

import numpy as np
import utils as rmu

from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat
from bokeh.models.layouts import Column, Row

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

        self.subjects = self.dtcwt_runner.subjects
        self.names = self.dtcwt_runner.names
        self.num_views = self.dtcwt_runner.num_views
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

            print 'Computing autocorrelation for subject', subject

            s_wavelets = self.wavelets[subject]
            day_pairs = zip(
                s_wavelets[:-1],
                s_wavelets[1:])

            for (p, (d1, d2)) in enumerate(day_pairs):
                for view in xrange(self.num_views):
                    (Yh1, Yl1) =  d1
                    (Yh2, Yl2) =  d2
                    Yh1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                    Yh2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
                    correlation = np.dot(
                        Yh1_mat[view].T, Yh2_mat[view])

                    self.correlation[subject][view].append(
                        correlation)

                    if self.save:
                        path = '_'.join([
                            'subject',
                            subject,
                            'view',
                            self.names[view], 
                            'periods',
                            '-'.join([str(p), str(p+1)])])
                        path = os.path.join(self.corr_dir, path)

                        with open(path, 'w') as f:
                            np.save(f, correlation)

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
            v = info[3]
            ps = [int(i) for i in info[5].split('-')]

            self.correlation[s][v][ps[0]] = m
        
    def _show(self):

        for (s, views) in self.correlation.items():
            for (view, periods) in enumerate(views):
                columns = [np.ravel(corr)[:,np.newaxis]
                           for corr in periods]
                timeline = np.hstack(columns)

                seaborn.heatmap(timeline)
                seaborn.plt.show()   
