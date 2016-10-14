import os

import numpy as np
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts

class SubperiodCorrelationRunner:

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
        self.num_subperiods = dtcwt_runner.num_sps
        default = lambda: [None for i in xrange(self.num_subperiods)]
        self.correlation = {s : SPUD(self.num_views, default=default)
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

        for (s, s_wavelets) in self.wavelets.items():
            spud = self.correlation[s]

            for (p, day) in s_wavelets:
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) =  subperiod[k[0]]
                        (Yh2, Yl2) =  subperiod[k[1]]
                        Y1_mat = rmu.get_sampled_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_sampled_wavelets(Yh2, Yl2)
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

        views = self.names[v[0]] + '-' + self.names[v[1]]
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
                    subperiods[i] = [None] * self.num_periods[l] 
                
        for (k, m) in correlation.items():
            info = k.split('_')
            s = info[1]
            v = [int(i) for i in info[3].split('-')]
            p = self.names2indices[info[5]]
            sp = int(info[7])

            self.correlation[s].get(v[0], v[1])[sp][p] = m

    def _show(self):

        for (s, spud) in self.correlation.items():
            for (k, subperiods) in spud.items():
                for (sp, periods) in enumerate(subperiods):
