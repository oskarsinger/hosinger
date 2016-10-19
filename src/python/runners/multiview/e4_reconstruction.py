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
        std=False):

        self.wavelets = dtcwt_runner.wavelets
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.num_views = dtcwt_runner.num_views
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

        default = lambda: [[] for i in xrange(self.num_subperiods)]

    def run(self):

        self._reconstruct()
        self._show()

    def _reconstruct(self):

        print 'Poop'

    def _show(self):

        print 'Poop'
