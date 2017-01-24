import os
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.shortcuts as dlstcts

from data.servers.batch import BatchServer as BS
from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts

class CMRawDataPlotRunner:

    def __init__(self,
        filepath,
        save_dir,
        period=24*3600,
        hsx=True,
        lsx=True,
        w=False,
        u=False):

        self.filepath = filepath
        self.period = period
        self.hsx = hsx
        self.lsx = lsx
        self.w = w
        self.u = u

        subdir = get_ts('_'.join([
            'period',
            str(period),
            'hsx',
            str(hsx),
            'lsx',
            str(lsx),
            'w',
            str(w),
            'u',
            str(u)]))
        
        self.save_dir = os.path.join(
            save_dir, subdir)

        os.mkdir(self.save_dir)

        self.valid_sympts = set()

        if self.hsx:
            self.valid_sympts.add('Hsx')

        if self.lsx:
            self.valid_sympts.add('Lsx')

        if self.w:
            self.valid_sympts.add('W')

        if self.u:
            self.valid_sympts.add('U')

        self.rate = 1.0 / (8.0 * 3600.0)
        self.window = self.rate * self.period
        self.loaders = dlstcts.get_cm_loaders_all_subjects(
            self.filepath)
        self.subjects = self.loaders.keys()

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.names = [dl.name()
                      for dl in sample_dls]
        self.servers = {s : [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}

    def run(self):

        ys = self._get_ys() 

        for (v, ys_v) in enumerate(ys):
            fig = plt.figure()

            for (i, (s, y)) in enumerate(ys_v.items()):
                ax = fig.add_subplot(
                    len(ys_v), 1, i+1)
                data_map = {s: (
                    self._get_x(y.shape[0], v, s),
                    y,
                    None)}

                plot_lines(
                    data_map,
                    'period',
                    'value',
                    '',
                    unit_name='Subject',
                    ax=ax)

            title = \
                'Mean value of view ' + \
                self.names[v] + \
                ' for period length ' + \
                str(self.period) + ' seconds'
            filename = '_'.join(title.split()) + '.png'
            path = os.path.join(
                self.save_dir, filename)

            fig.axes[0].set_title(title)
            plt.setp(
                [a.get_xticklabels() for a in fig.axes[:-1]],
                visible=False)
            plt.setp(
                [a.get_yticklabels() for a in fig.axes],
                visible=False)

            fig.savefig(path, format='png')
            sns.plt.clf()


    def _get_x(self, num_rows, v, s):

        return np.arange(num_rows).astype(float)[:,np.newaxis]

    def _get_ys(self):

        views = [{s : None for s in self.subjects}
                 for v in xrange(self.num_views)]

        for s in self.subjects:
            dss = self.servers[s]

            for (v, view) in enumerate(dss):

                name = self.names[v]
                data = view.get_data()
                float_num_periods = float(data.shape[0]) / self.window
                int_num_periods = int(float_num_periods)

                if float_num_periods - int_num_periods > 0:
                    int_num_periods += 1
                    full_length = int_num_periods * self.window
                    padding_l = full_length - data.shape[0]
                    padding = np.ones((padding_l, 1)) * np.nan
                    data = np.vstack([data, padding])

                reshaped = data.reshape(
                    (int_num_periods, self.window))
                means = np.mean(reshaped, axis=1)[:,np.newaxis]
                views[v][s] = np.copy(means)

        return views
