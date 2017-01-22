import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.shortcuts as dlstcts
import data.loaders.e4.utils as e4u

from data.servers.batch import BatchServer as BS
from lazyprojector import plot_lines
from drrobert.ts import get_dt_index

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        period=24*3600,
        missing=False,
        complete=False,
        std=False,
        avg_over_subjects=False,
        hsx=True,
        lsx=True,
        w=False,
        u=False):

        self.hdf5_path = hdf5_path
        self.period = period
        self.missing = missing
        self.complete = complete
        self.std = std
        self.avg_over_subjects = avg_over_subjects
        self.hsx = hsx
        self.lsx = lsx
        self.w = w
        self.u = u

        self.valid_sympts = set()

        if self.hsx:
            self.valid_sympts.add('Hsx')

        if self.lsx:
            self.valid_sympts.add('Lsx')

        if self.w:
            self.valid_sympts.add('W')

        if self.u:
            self.valid_sympts.add('U')

        self.name = 'Std' if self.std else 'Mean'

        self.loaders = dlstcts.get_e4_loaders_all_subjects(
            hdf5_path, None, False)
        self.servers = {s: [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.rates = [dl.get_status()['hertz']
                      for dl in sample_dls]
        self.names = [dl.name()
                      for dl in sample_dls]
        self.subjects = {s for s in self.servers.keys()[:4]
                         if e4u.get_symptom_status(s) in self.valid_sympts}

    def run(self):

        ys = self._get_ys()
        unit_name = 'Symptomatic?' \
            if self.avg_over_subjects else \
            None
        get_s_unit = lambda s: e4u.get_symptom_status(s) \
            if self.avg_over_subjects else \
            None
        s_units = {s : get_s_unit(s)
                   for s in self.subjects}

        for (v, ys_v) in enumerate(ys):
            title = \
                self.name + ' value of view ' + \
                self.names[v] + \
                ' for period length ' + \
                str(self.period) + ' seconds'

            if self.missing:
                title = 'Missing only ' + \
                    title[0].lower() + title[1:]
            elif self.complete:
                title = 'Complete only ' + \
                    title[0].lower() + title[1:]

            if self.avg_over_subjects:
                title = title + ' avg over subjects within symptom status'

            ax = plt.axes()
            data_map = {s : (self._get_x(y.shape[0], v, s), y, s_units[s])
                        for (s, y) in ys_v.items()}

            plot_lines(
                data_map,
                'period', 
                'value', 
                title,
                unit_name=unit_name,
                ax=ax)

            ax.get_figure().savefig(
                '_'.join(title.split()) + '.pdf',
                format='pdf')
            sns.plt.clf()

    def _get_x(self, num_rows, v, s):

        dt = self.loaders[s][v].get_status()['start_times'][0]
        dt_index_list = get_dt_index(
            num_rows, self.period, dt)
        dt_index_array = np.array(dt_index_list)

        return dt_index_array[:,np.newaxis]

    def _get_ys(self):

        views = [{s : None for s in self.subjects}
                 for v in xrange(self.num_views)]
        stat = np.std if self.std else np.mean

        for s in self.subjects:
            dss = self.servers[s]

            for (v, (r, view)) in enumerate(zip(self.rates, dss)):

                name = self.names[v]
                data = view.get_data()

                if name == 'TEMP':
                    data[data > 45] = 45
                elif name == 'EDA':
                    data[data > 30] = 30

                window = int(r * self.period)
                float_num_periods = float(data.shape[0]) / window
                int_num_periods = int(float_num_periods)

                if float_num_periods - int_num_periods > 0:
                    int_num_periods += 1
                    full_length = int_num_periods * window
                    padding_l = full_length - data.shape[0]
                    padding = np.ones((padding_l, 1)) * np.nan
                    data = np.vstack([data, padding])

                reshaped = data.reshape(
                    (int_num_periods, window))
                data_stats = stat(reshaped, axis=1)[:,np.newaxis]
                views[v][s] = np.copy(data_stats)

        return views
