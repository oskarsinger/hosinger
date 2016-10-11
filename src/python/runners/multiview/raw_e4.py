import seaborn

import numpy as np
import data.loaders.e4.shortcuts as dles

from data.pseudodata import MissingData as MD

class E4RawDataPlotRunner:

    def __init__(self, hdf5_path, num_periods=24):

        self.loaders = dles.get_e4_loaders_all_subjects(
            hdf5_path, None, True)
        self.num_views = len(self.loaders.values()[0])
        self.rates = [dl.get_status()['hertz']
                      for dl in self.loaders.values()[0]]

    def run(self):

        print 'Poop'

    def _plot_averaged_data(self):

        print 'Poop'

    def _get_averaged_data(self):

        views = [None] * self.num_views

        for (s, dls) in self.loaders.items():
            for view in dls: 

    def _get_t_tests(self):

        print 'Poop'
