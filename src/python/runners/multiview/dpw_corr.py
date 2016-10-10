import os

import numpy as np
import utils as rmu

from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat
from bokeh.models.layouts import Column, Row

class DayPairwiseCorrelationRunner:

    def __init__(self,
        wavelets,
        save_load_dir,
        save=False,
        load=False,
        show=False):

        self.wavelets = wavelets
        self.save = save
        self.load = load
        self.show = show

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.subjects = self.wavelets.subjects
        self.names = self.wavelets.names
        self.num_views = self.wavelets.num_views
        self.correlation = {s : [[] for i in xrange(self.num_views)]
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._load_wavelets()
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

            for (day1, day2) in day_pairs:
                for view in xrange(self.num_views):
                    #TODO: make sure this does subsampled wavelets instead
                    correlation = np.dot(
                        day1[view].T, day2[view])
                    self.correlation[subject][view].append(
                        autocorrelation)

    def _load(self):

        print 'Stuff'

    def _show(self):

        print 'Stuff'
