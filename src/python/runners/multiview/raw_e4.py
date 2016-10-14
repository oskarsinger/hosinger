import seaborn

import numpy as np
import pandas as pd
import data.loaders.e4.shortcuts as dles
import matplotlib.pyplot as plt

from data.pseudodata import MissingData as MD
from data.servers.batch import BatchServer as BS
from drrobert.arithmetic import get_running_avg as get_ra

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
        asymp = {'HRV15-0' + s
                 for s in ['06', '07', '13', '21', '24']}
        linestyles = ['--' if s in asymp else '-'
                      for s in self.servers.keys()]

        for (i, view) in enumerate(averages):
            ax = plt.axes()
            seaborn.pointplot(
                x='period', 
                y='value', 
                hue='subject',
                data=view,
                linestyles=linestyles,
                ax=ax)
            ax.set_title(
                'Mean value of view ' + 
                self.names[i] + 
                ' for period length ' + 
                str(self.period) + ' seconds')
            seaborn.plt.show()

    def _get_averages(self):

        views = [{s : None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]

        for (s, dss) in self.servers.items():
            print 'Computing averages for subject', s
            for (i, (r, view)) in enumerate(zip(self.rates, dss)):
                print '\tComputing averages for view', self.names[i]
                window = int(r * self.period)
                view_avg = []
                truncate = self.names[i] == 'TEMP'
                data = view.get_data()
                f_num_periods = float(data.shape[0]) / window
                i_num_periods = int(f_num_periods)

                if f_num_periods - i_num_periods > 0:
                    i_num_periods += 1

                for j in xrange(i_num_periods):
                    data_j = data[j * window: (j+1) * window]

                    if truncate:
                        data_j[data_j > 40] = 40

                    avg = np.mean(
                        data_j[np.logical_not(np.isnan(data_j))])

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
                l = l + [0] * (max_p - len(l))
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
