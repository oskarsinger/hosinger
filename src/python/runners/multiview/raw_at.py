import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.at.shortcuts as dlas

from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan

class ATRawDataPlotRunner:

    def __init__(self,
        tsv_path,
        period=24*3600,
        missing=False,
        complete=False,
        std=False):

        self.tsv_path = tsv_path
        self.period = period
        self.missing = missing
        self.complete = complete
        self.std = std
        self.name = 'Std' if self.std else 'Mean'

        self.loaders = dlas.get_at_loaders_all_subjects(
            self.tsv_path)
        self.servers = {s: [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.rates = [dl.get_status()['hertz']
                      for dl in sample_dls]
        self.names = [dl.name()
                      for dl in sample_dls]

        def run(self):

            averages = self._get
