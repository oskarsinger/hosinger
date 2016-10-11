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

            for (p, (Yhs, Yss)) in enumerate(s_wavelets):
                print 'Stuff'

