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
        std=False):

        self.tsv_path = tsv_path
        self.period = period
        self.std = std
        self.name = 'Std' if self.std else 'Mean'

        self.rate = 1.0/60
        self.window = int(self.rate * self.period)
        self.loaders = dlas.get_at_loaders_all_subjects(
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
            ax = plt.axes()

            sns.pointplot(
                x='period', 
                y='value', 
                hue='subject',
                data=view,
                ax=ax,
                legend=False)
            sns.plt.legend(
                bbox_to_anchor=(1, 1.05), 
                loc=2, 
                borderaxespad=0.)

            title = \
                self.name + ' value of view ' + \
                self.names[i] + \
                ' for period length ' + \
                str(self.period) + ' seconds'

            ax.set_title(title)
            ax.get_figure().savefig(
                '_'.join(title.split()) + '.pdf',
                format='pdf')
            sns.plt.clf()

    def _get_stats(self):
        
        views = [{s: None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]
        stat = np.std if self.std else np.mean
        dfs = [None] * self.num_views

        for (s, dss) in self.servers.items():
            for (i, view) in enumerate(dss):
                view_stat = []
                data = view.get_data()
                f_num_periods = float(data.shape[0]) / self.window
                i_num_periods = int(f_num_periods)

                if f_num_periods - i_num_periods > 0:
                    i_num_periods += 1

                for p in xrange(i_num_periods):
                    begin = p * self.window
                    end = begin + self.window

                    view_stat.append(stat(data[begin:end]))

                views[i][s] = view_stat

        for (i, view) in enumerate(views):
            periods = []
            subjects = []
            values = []

            for (s, l) in view.items():
                periods.extend(list(range(len(l))))
                subjects.extend([s] * len(l))
                values.extend(l)

            d = {
                'period': periods,
                'subject':subjects,
                'value': values}
            dfs[i] = pd.DataFrame(data=d)

        return dfs
