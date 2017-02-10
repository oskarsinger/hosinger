import os
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import data.loaders.shortcuts as dlstcts
import data.loaders.e4.utils as e4u

from data.servers.batch import BatchServer as BS
from data.servers.masks import Interp1DMask as I1DM
from drrobert.ts import get_dt_index
from drrobert.file_io import get_timestamped as get_ts

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        save_dir,
        period=24*3600,
        interpolate=False,
        hsx=True,
        lsx=True,
        w=False,
        u=False):

        self.hdf5_path = hdf5_path
        self.period = period
        self.interpolate = interpolate
        self.hsx = hsx
        self.lsx = lsx
        self.w = w
        self.u = u

        subdir = get_ts('_'.join([
            'period',
            str(period),
            'interpolate',
            str(interpolate),
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

        self.loaders = dlstcts.get_e4_loaders_all_subjects(
            hdf5_path, False)
        self.servers = {s: [BS(dl) for dl in dls]
                        for (s, dls) in self.loaders.items()}
        
        if self.interpolate:
            self.servers = {s : [I1DM(ds) for ds in dss]
                            for (s, dss) in self.servers.items()}

        sample_dls = self.loaders.values()[0]

        self.num_views = len(sample_dls)
        self.rates = [dl.get_status()['hertz']
                      for dl in sample_dls]
        self.names = [dl.name()
                      for dl in sample_dls]
        self.subjects = {s for s in self.servers.keys()
                         if e4u.get_symptom_status(s) in self.valid_sympts}

    def run(self):

        ys = self._get_ys()

        for (v, ys_v) in enumerate(ys):

            print  'Generating plots for view', self.names[v]

            fig = plt.figure()

            for (i, (s, y)) in enumerate(ys_v.items()):
                ax = fig.add_subplot(
                    len(ys_v), 1, i+1)

                self._plot_line(
                    s,
                    v,
                    y,
                    'time', 
                    '', 
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
            plt.clf()

    def _get_ys(self):

        views = [{s : None for s in self.subjects}
                 for v in xrange(self.num_views)]

        for s in self.subjects:
            dss = self.servers[s]

            for (v, (r, view)) in enumerate(zip(self.rates, dss)):

                name = self.names[v]
                data = view.get_data()
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
                means = np.mean(reshaped, axis=1)[:,np.newaxis]
                views[v][s] = np.copy(means)

        return views

    def _plot_line(self, s, v, ys, x_name, y_name, ax):

        dt = self.loaders[s][v].get_status()['start_times'][0]
        xs = np.array(get_dt_index(
            ys.shape[0], self.period, dt))

        ax.xaxis.set_major_locator(
            mdates.HourLocator(interval=24))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%d-th %H:%M'))
        ax.plot(xs, ys)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
