import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.shortcuts as dlstcts
import utils as rmu

from data.servers.batch import BatchServer as BS
from linal.utils.misc import get_non_nan
from lazyprojector import plot_lines

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        period=24*3600,
        missing=False,
        complete=False,
        std=False,
        avg_over_subjects=False):

        self.hdf5_path = hdf5_path
        self.period = period
        self.missing = missing
        self.complete = complete
        self.std = std
        self.avg_over_subjects = avg_over_subjects
        self.name = 'Std' if self.std else 'Mean'

        self.loaders = dlstcts.get_e4_loaders_all_subjects(
            hdf5_path, None, False)
        self.servers = {s: [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.rates = [dl.get_status()['hertz']
                      for dl in sample_dls]
        print 'self.rates', self.rates
        self.names = [dl.name()
                      for dl in sample_dls]
        # TODO: this may need to be altered according to complete and missing
        # TODO: how do I even define completeness now that the loader nan-pads?
        self.subjects = {s for s in self.servers.keys()
                         if rmu.get_symptom_status(s) not in {'W', 'U'}}

    def run(self):

        ys = self._get_ys()
        unit_name = 'Symptomatic?' \
            if self.avg_over_subjects else \
            None
        get_s_unit = lambda s: rmu.get_symptom_status(s) \
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
            data_map = {s : (np.arange(y.shape[0])[:,np.newaxis], y, s_units[s])
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

    def _get_ys(self):

        views = [{s : None for s in self.subjects}
                 for i in xrange(self.num_views)]
        stat = np.std if self.std else np.mean

        for s in self.subjects:
            dss = self.servers[s]

            print 'subject', s

            for (v, (r, view)) in enumerate(zip(self.rates, dss)):
                window = int(r * self.period)
                print 'view', v, 'rate', r, 'window', window

                truncate = self.names[v] in {'TEMP'}
                data = view.get_data()
                float_num_periods = float(data.shape[0]) / window
                int_num_periods = int(float_num_periods)
                print 'int_num_periods', int_num_periods

                if float_num_periods - int_num_periods > 0:
                    int_num_periods += 1
                    full_length = int_num_periods * window
                    padding_l = full_length - data.shape[0]
                    padding = np.ones((padding_l, 1)) * np.nan
                    data = np.vstack([data, padding])

                data = data.reshape(
                    (window, int_num_periods))
                print 'post reshape data.shape', data.shape

                if truncate:
                    data[data > 40] = 40

                views[v][s] = stat(data, axis=0)[:, np.newaxis]
                print 'views[v][s].shape', views[v][s].shape

        return views
