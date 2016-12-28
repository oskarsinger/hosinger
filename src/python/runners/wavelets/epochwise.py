import os
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.file_io import get_timestamped as get_ts

class EpochWiseTimeSeriesAnalysis:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        get_analysis_runner,
        boundaries,
        save=False,
        load=False,
        show=False):

        self.dtcwt_runner = dtcwt_runner
        self.get_analysis_runner = get_analysis_runner
        self.boundaries = boundaries

        self._init_dirs(
            save,
            load,
            show,
            save_load_dir)

        self.num_epochs = len(self.boundaries) + 1
        self.subjects = self.dtcwt_runner.subjects
        self.wavelets = self.dtcwt_runner.wavelets
        self.num_periods = dself.dtcwt_runner.num_periods
        self.save = save
        self.load = load
        self.show = show

        self.wavelets = dtcwt_runner.wavelets
        self.analysis_runners = [None] * self.num_epochs

    def _init_dirs(self,
        save,
        load,
        show,
        save_load_dir):

        if save:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('EpochWise')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

    def run(self):

        epochs = [{s : None for s in self.subjects}
                  for i in xrange(self.num_epochs)]

        for s in self.subjects:
            ps = self.wavelets[s]
            last = self.num_periods[s]
            ends = []

            for b in self.boundaries:
                if b >= last:
                    break

                ends.append(b)

            ends = ends + [last]
            begins = [0] + self.boundaries[:len(ends)-1]
            b_and_e = zip(begins, ends)

            for (i, (b, e)) in enumerate(b_and_e):
                epochs[i][s] = np.copy(ps[b:e])

        for (i, epoch) in enumerate(epochs):
            save_load_dir = os.path.join(
                self.save_load_dir,
                'Epoch' + str(i))

            os.mkdir(save_load_dir)

            self.analysis_runners[i] = self.get_analysis_runner(
                epoch,
                self.dtcwt_runner,
                save_load_dir,
                save=self.save,
                load=self.load,
                show=self.show)
