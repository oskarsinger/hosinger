import os
import h5py
import matplotlib

matplotlib.use('Cairo')
matplotlib.rcParams.update({'font.size': 9})

import numpy as np
import pandas as pd
import seaborn as sns
import utils as rmu
import matplotlib.pyplot as plt

from data.servers.batch import BatchServer as BS
from data.loaders.e4.utils import get_symptom_status
from drrobert.file_io import get_timestamped as get_ts
from linal.utils.misc import get_non_nan
from lazyprojector import plot_lines
from wavelets.dtcwt.oned import get_partial_reconstructions as get_pr

class DTCWTPartialReconstructionRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
        missing=False,
        complete=False,
        save=False,
        show=False,
        wavelets=None,
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

        # TODO: adapt to manually-entered wavelets as was done in vpw_corr
        self.wavelets = dtcwt_runner.wavelets if wavelets is None else wavelets
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

        num_freqs = {} 

        for v in xrange(self.num_views):
            num_freqs[v] = max(
                len(s_group[str(v)]) 
                for s_group in self.hdf5_repo.values())

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

        hdf5_path = os.path.join(
            self.save_load_dir, 'pr.hdf5')

        self.hdf5_repo = h5py.File(
            hdf5_path,
            'w' if save else 'r')
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
                        sp_v_prs = get_pr(
                            view[0], 
                            view[1],
                            self.biorthogonal,
                            self.qshift)

                        if view_stats[v] is None:
                            view_stats[v] = [None] * len(sp_v_prs)

                        for (f, pr) in enumerate(sp_v_prs):
                            max_p = int(factors[v] / 2**(f)) # - 1))
                            padding_l = max_p - pr.shape[0]

                            if padding_l > 0:
                                padding = np.array(
                                        [np.nan] * padding_l)[:,np.newaxis]
                                pr = np.vstack([pr, padding])

                            current = view_stats[v][f]

                            if current is None:
                                view_stats[v][f] = pr
                            else:
                                view_stats[v][f] = np.vstack(
                                    [current, pr])

            self._compute_completed_and_filtered(view_stats, s)

    def _show(self):

        num_freqs = [min([f, 7])
                     for f in self._get_num_freqs()]
        min_freqs = [max([3, f - 7])
                     for f in num_freqs] 

        for v in xrange(self.num_views):
            print 'Generating plots for view', v

            for f in xrange(min_freqs[v], num_freqs[v]):

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
                        1, self.num_plot_periods1, pp + 1)

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
                    '_'.join(title.split()) + '.png')

                fig.axes[0].set_title(title)
                plt.setp(
                    [a.get_yticklabels() for a in fig.axes[1:]],
                    visible=False)

		for a in fig.axes[1:]:
		    a.legend_.remove()

                fig.subplots_adjust(vspace=0)
                fig.savefig(path, format='png')
                sns.plt.clf()

    def _compute_completed_and_filtered(self, view_stats, s):

        print 'Padding partial reconstructions for subject', s

        s_unit = get_symptom_status(s) \
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

        stats = {s[-2:] : None for s in self.subjects}
        print 'type(v)', type(v), 'v', v
        print 'type(f)', type(f), 'f', f

        for (s, s_group) in self.hdf5_repo.items():
            print 's_group.values()', s_group.values()
            v_group = s_group.values()[v]
            f_group = v_group.values()[f]
            (p, v) = (f_group['p'][:,:], f_group['v'][:,:])
            u = f_group.attrs['u']

            stats[s] = (p, v, u)
        
        return stats

    def _save_stats(self, i, f, s, p, v, u):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        i_str = str(i)

        if i_str not in s_group:
            s_group.create_group(i_str)

        i_group = s_group[i_str]
        f_str = str(f)

        i_group.create_group(f_str)

        f_group = i_group[f_str]

        f_group.create_dataset('p', data=p)
        f_group.create_dataset('v', data=v)
        f_group.attrs['u'] = u

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
