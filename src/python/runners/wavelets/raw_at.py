import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import data.loaders.shortcuts as dlstcts

from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan
from lazyprojector import plot_lines

class ATRawDataPlotRunner:

    def __init__(self,
        tsv_path=None,
        period=24*3600,
        std=False):

        self.tsv_path = tsv_path
        self.period = period
        self.std = std
        self.name = 'Std' if self.std else 'Mean'

        self.rate = 1.0 / 60
        self.window = int(self.rate * self.period)

        if tsv_path is None:
            self.loaders = dlstcts.get_atr_loaders()
        else:
            self.loaders = dlstcts.get_ats_loaders_all_subjects(
                self.tsv_path)

        self.servers = {s: [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.names = [dl.name()
                      for dl in sample_dls]

    def run(self):

        averages = self._get_stats()
        
        for (i, view) in enumerate(averages):
            title = \
                self.name + ' value of view ' + \
                self.names[i] + \
                ' for period length ' + \
                str(self.period) + ' seconds'

            plot_lines(
                view,
                'period',
                'mean signal value',
                title
            ).get_figure().savefig(
                '_'.join(title.split()) + '.pdf',
                format='pdf')
            sns.plt.clf()

    def _get_stats(self):
        
        views = [{s: None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]
        stat = np.std if self.std else np.mean

        for (s, dss) in self.servers.items():
            for (i, view) in enumerate(dss):
                view_stat = []
                data = view.get_data()
                num_periods = int(float(data.shape[0]) / self.window)
                truncd = data[:num_periods * self.window,:]
                reshaped = truncd.reshape(
                    (num_periods,self.window))
                view_stat = stat(reshaped,axis=1)[:,np.newaxis]
                x = np.arange(num_periods)[:,np.newaxis]
                
                views[i][s] = (x, view_stat, None)

        return views
