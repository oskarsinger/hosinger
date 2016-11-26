import os
import matplotlib

matplotlib.use('Agg')

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

class DTCWTPartialReconstructionRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        missing=False,
        complete=False,
        save=False,
        load=False,
        show=False,
        avg=False):

        self.missing = False if avg else missing
        self.complete = True if avg else complete
        self.save = save
        self.load = load
        self.show = show
        self.avg = avg

        self._init_dirs(
            self.save,
            self.load,
            self.show,
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.servers = dtcwt_runner.servers
        self.g1a = dtcwt_runner.qshift['g1a']
        self.g1b = dtcwt_runner.qshift['g1b']
        self.g0a = dtcwt_runner.qshift['g0a']
        self.g0b = dtcwt_runner.qshift['g0b']
        self.g0o = dtcwt_runner.biorthogonal['g0o']
        self.g1o = dtcwt_runner.biorthogonal['g1o']
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

            model_dir = get_ts('DTCWTPRR')

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
                        sp_v_prs = self._get_view_sp_pr(
                            view[0], view[1])

                        if self.save:
                            self._save(
                                sp_v_prs,
                                s,
                                v,
                                p,
                                sp)

                        self.prs[s][p][sp][v] = sp_v_prs

    def _get_view_sp_pr(self, Yh, Yl):

        Lo_prev = np.copy(Yl)
        prs = [Lo_prev]

        for level in reversed(xrange(1, len(Yh))):
            Hi = wdtcwt.oned.c2q1d(Yh[level]) 
            Lo_filt = wdtcwt.filters.get_column_i_filtered(
                Lo_prev, self.g0b, self.g0a)
            Hi_filt = wdtcwt.filters.get_column_i_filtered(
                Hi, self.g1b, self.g1a)
            #doubled = _get_doubled_vector(Lo_prev)
            Lo_prev = Lo_filt + Hi_filt# - doubled

            Lo_n = Lo_prev.shape[0]
            Yh_n = Yh[level-1].shape[0]

            if not Lo_n == 2 * Yh_n:
                Lo_prev = Lo_prev[1:-1,:]

            prs.append(Lo_prev)

        Hi = wdtcwt.oned.c2q1d(Yh[0])
        Lo_filt = wdtcwt.filters.get_column_filtered(
            Lo_prev, self.g0o)
        Hi_filt = wdtcwt.filters.get_column_filtered(
            Hi, self.g1o)

        prs.append(Lo_filt + Hi_filt)# - Lo_prev)

        return list(reversed(prs))

    def _show(self):

        print 'Inside _show, getting stats'
        averages = self._get_stats()

        for (i, view) in enumerate(averages):
            print 'Inside _show, making plots for view', i

            for (f, freq) in enumerate(view):
                print 'Inside _show, making plots for frequency', f
                ax = plt.axes()

                condition  = 'Symptomatic?' if self.avg else 'Subject'
                unit = 'Subject' if self.avg else 'unit'

                print 'Inside _show, generating plot'
                sns.tsplot(
                    time='period', 
                    value='value', 
                    condition=condition,
                    unit=unit,
                    data=freq,
                    ax=ax)
                print 'Inside _show, plot generated'

                title = \
                    'View ' + \
                    self.names[i] + \
                    ' for ' + str(self.subperiod) + ' scnds' + \
                    ' rcnstrctd with dec. lvl' + \
                    ' 2^' + str(f)

                if self.avg:
                    title = 'Mean over sbjcts of ' + \
                            title[0].lower() + title[1:]

                if self.missing:
                    title = 'Missing only ' + \
                        title[0].lower() + title[1:]
                elif self.complete:
                    title = 'Complete only ' + \
                        title[0].lower() + title[1:]

                ax.set_title(title)
                path = os.path.join(
                    self.plot_dir,
                    '_'.join(title.split()) + '.pdf')
                print 'Inside _show, saving plot'
                ax.get_figure().savefig(
                    path,
                    format='pdf')
                print 'Inside _show, clearing figure'
                sns.plt.clf()

    def _get_stats(self):

        view_stats = [{s[-2:] : [] for s in self.subjects}
                      for i in xrange(self.num_views)]

        for (s, periods) in self.prs.items():
            print 'Inside _get_stats, subject', s
            s = s[-2:]
            sample_views = periods[0][0]

            for (v, prs) in enumerate(sample_views):
                view_stats[v][s] = [None for f in xrange(len(prs))]

            for (p, subperiods) in enumerate(periods):
                print 'Inside _get_stats, period', p
                for (sp, views) in enumerate(subperiods):
                    print 'Inside _get_stats, subperiod', sp
                    for (v, prs) in enumerate(views):
                        print 'Inside _get_stats, view', v
                        for (f, pr) in enumerate(prs):
                            print 'Inside _get_stats, frequency', f
                            current = view_stats[v][s][f]

                            if current is None:
                                view_stats[v][s][f] = pr
                            else:
                                view_stats[v][s][f] = np.vstack(
                                    [current, pr])

        return self._get_completed_and_filtered(view_stats)

    def _get_completed_and_filtered(self, view_stats):

        dfs = [[None] * len(view.values()[0])
               for view in view_stats]
        unit_key = 'Symptomatic?' if self.avg else 'unit'

        for (i, view) in enumerate(view_stats):
            print 'Inside _get_completed_and_filtered, view', i
            num_lists = len(view.values()[0])
            periods = [[] for f in xrange(num_lists)]
            subjects = [[] for f in xrange(num_lists)]
            values = [None for f in xrange(num_lists)]
            units = [[] for f in xrange(num_lists)]
            max_ps = [max(len(l[f]) for l in view.values())
                      for f in xrange(num_lists)]

            for (s, freqs) in view.items():
                print 'Inside _get_completed_and_filtered, subject', s
                for (f, freq) in enumerate(freqs):
                    print 'Inside _get_completed_and_filtered, frequency', f
                    max_p = max_ps[f]
                    l_freq = freq.shape[0]
                    padding = np.array([None] * (max_p - l_freq))[:,np.newaxis]
                    freq_list = np.vstack([freq, padding])
                    s_periods = list(range(max_p))
                    s_subjects = [s] * max_p
                    s_units = None

                    if self.avg:
                        status = rmu.get_symptom_status(s)
                        s_units = [status] * max_p
                    else:
                        s_units = [1] * max_p

                    first = self.missing and l_freq < max_p
                    second = self.complete and l_freq == max_p
                    third = not (self.missing or self.complete)

                    if first or second or third:
                        periods[f].extend(s_periods)
                        subjects[f].extend(s_subjects)
                        units[f].extend(s_units)

                        if values[f] is None:
                            values[f] = freq
                        else:
                            values[f] = np.vstack(
                                [values[f], freq])

            for f in xrange(num_lists):
                print 'Inside _get_completed_and_filtered, getting df for frequency', f
                d = {
                    'period': periods[f],
                    'Subject': subjects[f], 
                    'value': values[f],
                    unit_key: units[f]}
                dfs[i][f] = pd.DataFrame(data=d)

        return dfs

    def _load(self):

        print 'Loading partial reconstructions'
        for fn in os.listdir(self.pr_dir):
            info = fn.split('_')
            s = info[1]
            v = int(info[3])
            p = int(info[5])
            sp = int(info[7])
            path = os.path.join(self.pr_dir, fn)
            prs = None

            with open(path) as f:
                loaded = {int(h_fn.split('_')[1]) : a
                          for (h_fn, a) in np.load(f).items()}
                prs = [loaded[i] 
                       for i in xrange(len(loaded))]

            self.prs[s][p][sp][v] = prs

    def _save(self, prs, s, v, p, sp):

        fname = '_'.join([
            'subject', s,
            'view', str(v),
            'period', str(p),
            'subperiod', str(sp)])
        path = os.path.join(self.pr_dir, fname)

        with open(path, 'w') as f:
            np.savez(f, *prs)

def _get_doubled_vector(v):

    n = v.shape[0]
    doubled = np.zeros((n*2, 1))
    doubled[0::2,:] = np.copy(v)
    doubled[1::2,:] = np.copy(v)

    return doubled
