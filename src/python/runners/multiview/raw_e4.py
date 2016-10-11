import seaborn

import numpy as np
import data.loaders.e4.shortcuts as dles

class E4RawDataPlotRunner:

    def __init__(self, hdf5_path):

        self.loaders = dles.get_e4_loaders_all_subjects(
            hdf5_path, None, False)

    def run(self):
        print 'Poop'

    def _get_averaged_data(self):

        print 'Poop'
