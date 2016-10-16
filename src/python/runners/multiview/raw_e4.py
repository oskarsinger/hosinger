import seaborn as sns

import numpy as np
import pandas as pd
import data.loaders.e4.shortcuts as dles
import matplotlib.pyplot as plt

from data.pseudodata import MissingData as MD
from data.servers.batch import BatchServer as BS
from drrobert.arithmetic import get_running_avg as get_ra
from linal.utils.misc import get_non_nan

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        period=24*3600):

        self.hdf5_path = hdf5_path
        self.period = period

        self.loaders = dles.get_e4_loaders_all_subjects(
            hdf5_path, None, False)
        self.servers = {s: [BS(dl) for dl in dls]
            for (s, dls) in self.loaders.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.rates = [dl.get_status()['hertz']
                      for dl in sample_dls]
        self.names = [dl.name()
                      for dl in sample_dls]

    def run(self):

        self._plot_averages()

    def _plot_averages(self):

        averages = self._get_averages()
        asymp = {'06', '07', '13', '21', '24'}
        linestyles = ['--' if s in asymp else '-'
                      for s in self.servers.keys()]

        for (i, view) in enumerate(averages):
            ax = plt.axes()
            sns.pointplot(
                x='period', 
                y='value', 
                hue='subject',
                data=view,
                linestyles=linestyles,
                ax=ax,
                legend=False)
            sns.plt.legend(
                bbox_to_anchor=(1, 1.05), 
                loc=2, 
                borderaxespad=0.)
            title = \
                'Mean value of view ' + \
                self.names[i] + \
                ' for period length ' + \
                str(self.period) + ' seconds'
            ax.set_title(title)

            ax.get_figure().savefig(
                '_'.join(title.split()) + '.png')

            sns.plt.clf()

    def _get_averages(self):

        views = [{s[-2:] : None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]

        for (s, dss) in self.servers.items():
            s = s[-2:]
            for (i, (r, view)) in enumerate(zip(self.rates, dss)):
                window = int(r * self.period)
                view_avg = []
                truncate = self.names[i] == 'TEMP'
                data = view.get_data()
                f_num_periods = float(data.shape[0]) / window
                i_num_periods = int(f_num_periods)

                if f_num_periods - i_num_periods > 0:
                    i_num_periods += 1

                for p in xrange(i_num_periods):
                    data_p = data[p * window : (p+1) * window]
                    data_p = get_non_nan(data_p)

                    if truncate:
                        data_p[data_p > 40] = 40

                    avg = np.mean(data_p)

                    view_avg.append(avg)

                views[i][s] = view_avg

        dfs = [None] * self.num_views

        for (i, view) in enumerate(views):
            periods = []
            subjects = []
            values = []

            max_p = max(
                [len(l) for l in view.values()])

            for (s, l) in view.items():
                l = l + [None] * (max_p - len(l))
                periods.extend(list(range(max_p)))
                subjects.extend([s] * max_p)
                values.extend(l)

            d = {
                'period': periods,
                'subject': subjects, 
                'value': values}
            dfs[i] = pd.DataFrame(data=d)

        return dfs

    def _get_t_tests(self):

        print 'Poop'
