import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt

from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan

class E4DTCWTPartialReconstructionRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        missing=False,
        complete=False,
        std=False,
        save=False,
        load=False,
        show=False):

        # TODO: init directories
        self.save = save
        self.load = load
        self.show = show

        # TODO: should this be by periods or subperiods?
        self.wavelets = dtcwt_runner.wavelets
        self.g1a = dtcwt_runner.qshift['g1a']
        self.g1b = dtcwt_runner.qshift['g1b']
        self.g0a = dtcwt_runner.qshift['g0a']
        self.g0b = dtcwt_runner.qshift['g0b']
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

    def run(self):

        self._reconstruct()
        self._show()

    def _reconstruct(self):

        self.recons = rmu.get_wavelet_storage(
            self.num_views,
            self.num_subperiods,
            self.num_periods,
            self.subjects)

        for (s, periods) in self.wavelets.items():
            for (p, subperiods) in enumerate(periods):
                for (sp, views) in enumerate(subperiods):
                    for (v, view) in enumerate(views):
                        (His, Lo) = self._get_reconstructed_view_sp(
                            view[0], view[1])

                        self.recons[s][p][sp][v][0] = His
                        self.recons[s][p][sp][v][1] = Lo

    def _get_reconstructed_view_sp(self, Yh, Yl):

        Lo = np.copy(Yl)
        His = [wdtcwt.oned.c2q1d(Y) for Y in Yh]
        Lo_filt = dtcwt.filters.get_column_i_filtered(
            Lo, self.g0b, self.g0a)
        Hi_filts = [dtcwt.filters.get_column_i_filtered(
                        Hi, self.g1b, self.g1a)
                    for Hi in His]

        return (Hi_filts, Lo_filt)

    def _show(self):

        print 'Poop'