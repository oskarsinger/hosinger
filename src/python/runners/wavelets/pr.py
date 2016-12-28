import os
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import utils as rmu
import matplotlib.pyplot as plt
import wavelets.dtcwt as wdtcwt

from data.servers.batch import BatchServer as BS
from drrobert.file_io import get_timestamped as get_ts
from linal.utils.misc import get_non_nan
from lazyprojector import plot_lines

class DTCWTPartialReconstructionRunner:

    def __init__(self,
        #wavelets,
        dtcwt_runner,
        save_load_dir,
        missing=False,
        complete=False,
        save=False,
        show=False,
        avg_over_periods=False,
        avg_over_subjects=False,
        num_plot_periods=None):

        self.missing = False if avg_over_subjects else missing
        self.complete = True if avg_over_subjects else complete
        self.save = save
        self.show = show
        self.avg_over_periods = avg_over_periods
        self.avg_over_subjects = avg_over_subjects

        self._init_dirs(
            self.save,
            self.show,
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets
        self.biorthogonal = dtcwt_runner.biorthogonal
        self.qshift = dtcwt_runner.qshift
        self.subjects = dtcwt_runner.subjects
        self.names = dtcwt_runner.names
        self.num_views = dtcwt_runner.num_views
        self.period = dtcwt_runner.period
        self.subperiod = dtcwt_runner.subperiod
        self.rates = dtcwt_runner.rates
        self.num_periods = dtcwt_runner.num_periods
        self.num_subperiods = dtcwt_runner.num_sps

        if num_plot_periods is None:
            num_plot_periods = self.num_subperiods

        if not self.avg_over_periods:
            num_plot_periods *= max(self.num_periods)

        self.num_plot_periods = num_plot_periods
        self.prs = rmu.get_wavelet_storage(
            self.num_views,
            self.num_subperiods,
            self.num_periods,
            self.subjects)

    def run(self):

        if self.show:
            self._show()
        else:
            self._compute()

    def _get_num_freqs(self):

        fns = os.listdir(self.stat_dir)
        split = [fn.split('_') for fn in fns]
        pairs = {(int(s[3]), s[5]) for s in split}
        num_freqs = [0 for v in xrange(self.num_views)]

        for (v, f) in pairs:
            num_freqs[v] += 1

        return num_freqs

    def _init_dirs(self,
        save,
        show,
        save_load_dir):

        if save:
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
        self.stat_dir = rmu.init_dir(
            'stats',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir)

    def _compute(self):

        for (s, periods) in self.wavelets.items():
            print 'Computing partial reconstructions for subject', s
            s = s[-2:]
            view_stats = [None] * self.num_views
            factors = [r * self.subperiod for r in self.rates]

            for (p, subperiods) in enumerate(periods):
                for (sp, views) in enumerate(subperiods):
                    for (v, view) in enumerate(views):
                        sp_v_prs = self._get_view_sp_pr(
                            view[0], view[1])

                        if view_stats[v] is None:
                            view_stats[v] = [None] * len(sp_v_prs)

                        for (f, pr) in enumerate(sp_v_prs):
                            max_p = int(factors[v] / 2**(f)) # - 1))
                            padding_l = max_p - pr.shape[0]

                            if padding_l > 0:
                                padding = np.array([np.nan] * padding_l)
                                pr = np.vstack([pr, padding_l])

                            current = view_stats[v][f]

                            if current is None:
                                view_stats[v][f] = pr
                            else:
                                view_stats[v][f] = np.vstack(
                                    [current, pr])

            self._compute_completed_and_filtered(view_stats, s)

    def _get_view_sp_pr(self, Yh, Yl):

        prs = []
        Ylz = np.zeros_like(Yl)
        mask = np.zeros((1,len(Yh)))

        for i in xrange(len(Yh)):
            mask = mask * 0
            mask[0,i] = 1
            pr = wdtcwt.oned.dtwaveifm(
                Ylz,
                Yh,
                self.biorthogonal,
                self.qshift,
                gain_mask=mask)
            
            prs.append(pr)

        Yl_pr = wdtcwt.oned.dtwaveifm(
            Yl,
            Yh,
            self.biorthogonal,
            self.qshift,
            gain_mask=mask * 0)
        prs.append(Yl_pr)

        return prs

    def _show(self):

        num_freqs = [min(f, 5)
                     for f in self._get_num_freqs()]

        for v in xrange(self.num_views):
            print 'Generating plots for view', v

            for f in xrange(2, num_freqs[v]):

                print 'Retrieving data for freq', f

                freq = self._load_stats(v, f)
                fig = plt.figure()
                title = \
                    'View ' + \
                    self.names[v] + \
                    ' rcnstrctd with decmtn' + \
                    ' 2^' + str(f)

                if self.avg_over_subjects:
                    title = 'Mean over sbjcts of ' + \
                            title[0].lower() + title[1:]

                if self.missing:
                    title = 'Missing only ' + \
                        title[0].lower() + title[1:]
                elif self.complete:
                    title = 'Complete only ' + \
                        title[0].lower() + title[1:]
            
                if self.avg_over_periods:
                    freq = self._get_avg_period(freq)

                for pp in xrange(self.num_plot_periods):
                    ax = fig.add_subplot(
                        self.num_plot_periods, 1, pp + 1)

                    pp_freq = self._get_pp_freq(freq, pp)
                    unit_name = 'Symptomatic?' \
                        if self.avg_over_subjects else \
                        None

                    print 'Generating plot for plot period', str(pp)

                    plot_lines(
                        pp_freq,
                        'period',
                        'value',
                        '',
                        unit_name=unit_name,
                        ax=ax)

                path = os.path.join(
                    self.plot_dir,
                    '_'.join(title.split()) + '.pdf')

                fix.axes[0].set_title(title)
                plt.setp(
                    [a.get_xticklabels() for a in fig.axes[:-1]], 
                    visible=False)
                fig.subplots_adjust(hspace=0)
                fig.savefig(path, format='pdf')
                sns.plt.clf()

    def _compute_completed_and_filtered(self, view_stats, s):

        print 'Padding partial reconstructions for subject', s

        unit_key = 'Symptomatic?' \
            if self.avg_over_subjects else \
            'unit'
        s_unit = rmu.get_symptom_status(s) \
            if self.avg_over_subjects else \
            None

        for (v, freqs) in enumerate(view_stats):
            for (f, freq) in enumerate(freqs):
                self._save_stats(
                    v, f, s,
                    np.arange(freq.shape[0])[:,np.newaxis],
                    freq,
                    s_unit)

    def _load_stats(self, v, f):

        is_v = lambda fn: 'view_' + str(v) in fn
        is_f = lambda fn: 'frequency_' + str(f) in fn
        fns = os.listdir(self.stat_dir)
        vf_fns = [fn for fn in fns
                  if is_v(fn) and is_f(fn)]
        stats = {s[-2:] : None for s in self.subjects}

        for fn in vf_fns:
            info = fn.split('_')
            s = info[1]
            path = os.path.join(self.stat_dir, fn)

            if s in self.subjects:
                with open(path) as f:
                    loaded = {int(h_fn.split('_')[1]) : a
                              for (h_fn, a) in np.load(f).items()}
                    x = loaded[0]
                    y = loaded[1]
                    u = loaded[2]
                    u = None if u.ndim == 0 else u[:,np.newaxis]
                    stats[s] = (x, y, u)
        
        return stats

    def _save_stats(self, i, f, s, p, v, u):

        fname = '_'.join([
            'subject', s,
            'view', str(i),
            'frequency', str(f)])
        path = os.path.join(self.stat_dir, fname)

        with open(path, 'w') as f:
            np.savez(f, *[p, v, u])

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

        self.num_freqs = [len(self.prs.values()[0][0][0][i])
                          for i in xrange(self.num_views)]

    def _save(self, prs, s, v, p, sp):

        fname = '_'.join([
            'subject', s,
            'view', str(v),
            'period', str(p),
            'subperiod', str(sp)])
        path = os.path.join(self.pr_dir, fname)

        with open(path, 'w') as f:
            np.savez(f, *prs)

    def _get_pp_freq(self, freq, pp):
        
        pp_freq = {}

        for (s, (x, y, u)) in freq.items():
            pp_length = int(x.shape[0] / self.num_plot_periods)
            begin = pp * pp_length
            end = begin + pp_length

            pp_x = np.copy(x[begin:end])
            pp_y = np.copy(y[begin:end])

            pp_freq[s] = (pp_x, pp_y, u)

        return pp_freq

    def _get_avg_period(self, freq):

        avg_freq = {}

        for (s, (x, y, u)) in freq.items():
            period_length = int(y.shape[0] / self.num_periods[s])
            truncd_length = period_length * self.num_periods[s]
            period_rows = np.reshape(
                y[:truncd_length],
                (self.num_periods[s], period_length))
            avg_y = np.mean(period_rows, axis=0)[:,np.newaxis]
            avg_x = x[:avg_y.shape[0],:]
            avg_freq[s] = (avg_x, avg_y, u)

        return avg_freq
