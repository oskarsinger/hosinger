import seaborn

import numpy as np
import pandas as pd
import data.loaders.e4.shortcuts as dles

from data.pseudodata import MissingData as MD
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import get_running_avg as get_ra

class E4RawDataPlotRunner:

    def __init__(self, 
        hdf5_path,
        period=24*3600):

        self.hdf5_path = hdf5_path
        self.period = period

        self.loaders = dles.get_e4_loaders_all_subjects(
            hdf5_path, None, True)
        self.servers = {s: [M2M(dl, 1) for dl in dls]
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
            seaborn.pointplot(
                x='period', 
                y='value', 
                hue='subject',
                data=view,
                linestyles=linestyles)
            seaborn.plt.show()

    def _get_averages(self):

        views = [{s : None for s in self.servers.keys()}
                 for i in xrange(self.num_views)]

        for (s, dss) in self.servers.items():
            print 'Computing averages for subject', s
            for (i, (r, view)) in enumerate(zip(self.rates, dss)):
                print '\tComputing averages for view', self.names[i]
                view_avg = [0]
                num_real_data = 0
                truncate = self.names[i] == 'TEMP'

                while not view.finished():
                    data = view.get_data()
                    threshold = self.period * len(view_avg)
                    quantity = view.rows() / r

                    if quantity >= threshold:
                        print '\t\tComputing average for period', len(view_avg)
                        view_avg.append(0)
                        num_real_data = 0

                    if not isinstance(data, MD):
                        num_real_data += 1
                        data = data[0][0]
                        
                        if truncate and data > 40:
                            data = 40

                        view_avg[-1] = get_ra(
                            view_avg[-1], data, num_real_data)

                views[i][s] = view_avg

        for (i, view) in enumerate(views):
            periods = []
            subjects = []
            values = []

            for (s, l) in view.items():
                periods.extend(list(xrange(len(l))))
                subjects.extend([s] * len(l))
                values.extend(l)

            d = {
                'period': periods,
                'subject': subjects, 
                'value': values}
            views[i] = pd.DataFrame(data=d)

            """
            max_periods = max(
                [len(l) for l in view.values()])
            extended = [l + [0] * (max_periods - len(l))
                        for l in view.values()]
            views[i] = [np.array(l) for l in extended]
            """

        return views

    def _get_t_tests(self):

        print 'Poop'
