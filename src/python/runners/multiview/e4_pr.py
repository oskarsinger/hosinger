import os

import numpy as np
import pandas as pd
import seaborn as sns
import utils as rmu
import matplotlib.pyplot as plt
import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt

from data.servers.batch import BatchServer as BS
from drrobert.file_io import get_timestamped as get_ts
from linal.utils.misc import get_non_nan

class E4DTCWTPartialReconstructionRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        missing=False,
        complete=False,
        std=False,
        save=False,
        load=False,
        show=False):

        self.std = std
        self.save = save
        self.load = load
        self.show = show

        self._init_dirs(
            self.save,
            self.load,
            self.show,
            save_load_dir)

        self.name = 'Std' if self.std else 'Mean'
        self.wavelets = dtcwt_runner.wavelets
        self.g1a = dtcwt_runner.qshift['g1a']
        self.g1b = dtcwt_runner.qshift['g1b']
        self.g0a = dtcwt_runner.qshift['g0a']
        self.g0b = dtcwt_runner.qshift['g0b']
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.num_views = dtcwt_runner.num_views
        self.period = dtcwt_runner.period
        self.subperiod = dtcwt_runner.subperiod
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

        self.prs = rmu.get_wavelet_storage(
            self.num_views,
            self.num_subperiods,
            self.num_periods,
            self.subjects)

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show()

    def _init_dirs(self,
        save,
        load,
        show,
        save_load_dir):

        if (show or save) and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('E4DTCWTPRR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.pr_dir = rmu.init_dir(
            'pr',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir)

    def _compute(self):

        for (s, periods) in self.wavelets.items():
            for (p, subperiods) in enumerate(periods):
                for (sp, views) in enumerate(subperiods):
                    for (v, view) in enumerate(views):
                        sp_v_prs = self._get_reconstructed_view_sp(
                            view[0], view[1])

                        if self.save:
                            self._save(
                                sp_v_prs,
                                s,
                                v,
                                p,
                                sp)

                        self.prs[s][p][sp][v] = sp_v_prs

    def _get_reconstructed_view_sp(self, Yh, Yl):

        Lo_prev = np.copy(Yl)
        prs = [Lo_prev]

        for level in reversed(xrange(1, len(Yh))):
            Hi = c2q1d(Yh[level]) 
            Lo_filt = filters.get_column_i_filtered(
                Lo_prev, g0b, g0a)
            Hi_filt = filters.get_column_i_filtered(
                Hi, g1b, g1a)
            Lo_prev = Lo_filt + Hi_filt - Lo_prev

            prs.append(Lo_prev)

        Hi = c2q1d(Yh[0])
        Lo_filt = filters.get_column_filtered(
            Lo_prev, g0o)
        Hi_filt = filters.get_column_filtered(
            Hi, g1o)

        prs.append(Lo_filt + Hi_filt - Lo_prev)

        return prs

    def _show(self):

        averages = self._get_stats()
        asymp = {'06', '07', '13', '21', '24'}
        linestyles = ['--' if s[-2:] in asymp else '-'
                      for s in self.servers.keys()]

        for (i, view) in enumerate(averages):
            for (f, freq) in enumerate(view):
                ax = plt.axes()

                sns.pointplot(
                    x='period', 
                    y='value', 
                    hue='subject',
                    data=freq,
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
                    str(self.subperiod) + ' seconds' + \
                    ' reconstructed with decimation level' + \
                    ' 2^' + str(f)

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

        views = [{s[-2:] : None for s in self.subjects}
                 for i in xrange(self.num_views)]
        stat = np.std if self.std else np.mean

        for (s, view_list) in self.wavelets.items():
            s = s[-2:]

            for (v, periods) in enumerate(view_list):
                (Yh, Yl) = periods[0][0]
                num_freqs = len(Yh) + 1
                view_stat = [[] for i in xrange(num_freqs)]

                for (p, subperiods) in enumerate(periods):
                    for (sp, prs) in enumerate(subperiods):
                        pr_stats = [stat(pr) for pr in prs]

                        for (f, pr) in enumerate(pr_stats):
                            view_stat[f].append(pr)

                views[v][s] = view_stat

        return self._get_completed_and_filtered(views)

    def _get_completed_and_filtered(self, view_stats):

        dfs = [[None] * len(view)
               for view in view_stats]

        for (i, view) in enumerate(view_stats):
            max_p = max(
                [len(l[0]) for l in view.values()])
            num_lists = len(view.values()[0])
            periods = [[] for f in xrange(num_lists)]
            subjects = [[] for f in xrange(num_lists)]
            values = [[] for f in xrange(num_lists)]

            for (s, freqs) in view.items():
                for (f, freq) in enumerate(freqs):
                    ll = len(l)
                    l = l + [None] * (max_p - ll)
                    s_periods = list(range(max_p))
                    s_subjects = [s] * max_p

                    if self.missing:
                        if ll < max_p:
                            periods[f].extend(s_periods)
                            subjects[f].extend(s_subjects)
                            values[f].extend(l)
                    elif self.complete:
                        if ll == max_p:
                            periods[f].extend(s_periods)
                            subjects[f].extend(s_subjects)
                            values[f].extend(l)
                    else:
                        periods[f].extend(s_periods)
                        subjects[f].extend(s_subjects)
                        values[f].extend(l)

            for f in len(view.values[0]):
                d = {
                    'period': periods[f],
                    'subject': subjects[f], 
                    'value': values[f]}
                dfs[i][f] = pd.DataFrame(data=d)

        return dfs

    def _load(self):

        for fn in os.listdir(self.pr_dir):
            path = os.path.join(self.pr_dir, fn)
            prs = None

            with open(path) as f:
                loaded = np.load(f)
                prs = [loaded[i] 
                       for i in xrange(len(loaded))]

            info = fn.split('_')
            s = info[1]
            v = int(info[3])
            p = int(info[5])
            sp = int(info[7])

            self.prs[s][v][p][sp] = prs

    def _save(self, prs, s, v, p, sp):

        path = '_'.join([
            'subject', s,
            'view', str(v),
            'period', str(p),
            'subperiod', str(sp)])
        path = os.path.join(self.pr_dir, path)

        with open(path, 'w') as f:
            np.savez(f, prs)
