import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.e4.shortcuts as dles

from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        period=24*3600,
        missing=False,
        complete=False,
        std=False):

        self.hdf5_path = hdf5_path
        self.period = period
        self.missing = missing
        self.complete = complete
        self.std = std
        self.name = 'Std' if self.std else 'Mean'

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

        averages = self._get_completed_and_filtered(
            self._get_stats())
        asymp = {'06', '07', '13', '21', '24'}
        linestyles = ['--' if s[-2:] in asymp else '-'
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
                self.name + ' value of view ' + \
                self.names[i] + \
                ' for period length ' + \
                str(self.period) + ' seconds'

            if self.missing:
                title = 'Missing only ' + \
                    title[0].lower() + title[1:]
            elif self.complete:
                title = 'Complete only ' + \
                    title[0].lower() + title[1:]

            ax.set_title(title)
            ax.get_figure().savefig(
                '_'.join(title.split()) + '.png')
            sns.plt.clf()

    def _get_stats(self):

        views = [{s[-2:] : None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]
        stat = np.std if self.std else np.mean

        for (s, dss) in self.servers.items():
            s = s[-2:]

            for (i, (r, view)) in enumerate(zip(self.rates, dss)):
                window = int(r * self.period)
                view_stat = []
                truncate = self.names[i] == 'TEMP'
                data = view.get_data()
                f_num_periods = float(data.shape[0]) / window
                i_num_periods = int(f_num_periods)

                if f_num_periods - i_num_periods > 0:
                    i_num_periods += 1

                for p in xrange(i_num_periods):
                    data_p = get_non_nan(
                        data[p * window : (p+1) * window])

                    if truncate:
                        data_p[data_p > 40] = 40

                    view_stat.append(stat(data_p))

                views[i][s] = view_stat

        return views

    def _get_completed_and_filtered(self, view_stats):

        dfs = [None] * self.num_views

        for (i, view) in enumerate(view_stats):
            periods = []
            subjects = []
            values = []

            max_p = max(
                [len(l) for l in view.values()])

            for (s, l) in view.items():
                ll = len(l)
                l = l + [None] * (max_p - ll)
                s_periods = list(range(max_p))
                s_subjects = [s] * max_p

                if self.missing:
                    if ll < max_p:
                        periods.extend(s_periods)
                        subjects.extend(s_subjects)
                        values.extend(l)
                elif self.complete:
                    if ll == max_p:
                        periods.extend(s_periods)
                        subjects.extend(s_subjects)
                        values.extend(l)
                else:
                    periods.extend(s_periods)
                    subjects.extend(s_subjects)
                    values.extend(l)

            d = {
                'period': periods,
                'subject': subjects, 
                'value': values}
            dfs[i] = pd.DataFrame(data=d)

        return dfs
