import os
import h5py
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.stats import get_cca_vecs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.file_io import init_dir
from drrobert.misc import unzip
from drrobert.ts import get_dt_index as get_dti
from lazyprojector import plot_matrix_heat
from exploratory.mvc.utils import get_matched_dims
from math import log, ceil

class NTPTViewPairwiseCCA:

    def __init__(self,
        servers,
        save_load_dir,
        num_subperiods=1,
        cov_analysis=True,
        clock_time=False,
        show=False):

        self.servers = servers
        self.num_subperiods = num_subperiods
        # TODO: implement optional cov_analysis like in Al's script
        self.cov_analysis = cov_analysis
        # TODO: implement clock time x axis on heat plot
        self.clock_time = clock_time
        self.show = show

        self.subjects = self.servers.keys()
        self.cols = [ds.cols() for ds in self.servers.values()[0]]
        self.subperiod = int(24.0 * 3600.0 / self.num_subperiods)
        self.loaders = {s : [ds.get_status()['data_loader'] for ds in dss]
                        for (s, dss) in self.servers.items()}
        self.names = [dl.name() for dl in self.loaders.values()[0]]
        self.num_views = len(self.servers.values()[0])
        self.num_periods = {s : int(servers[0].num_batches / self.num_subperiods)
                            for (s, servers) in self.servers.items()}

        self._init_dirs(save_load_dir)

        self.cca = {s : SPUD(
                        self.num_views, 
                        no_double=True)
                    for s in self.subjects}

    def run(self):

        if self.show:
            self._load()
            self._show()
        else:
            self._compute()

    def _init_dirs(self, save_load_dir):

        if self.show:
            self.save_load_dir = save_load_dir
        else:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('EBNTPFVPWCCA')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)

        hdf5_path = os.path.join(
            self.save_load_dir, 'ccas')
        self.hdf5_repo = h5py.File(
            hdf5_path, 
            'r' if self.show else 'w')
        self.plot_dir = init_dir(
            'plots',
            self.show,
            self.save_load_dir) 

    def _compute(self):

        tls = {s : [[None] * self.num_subperiods
                    for i in xrange(self.num_views)]
               for s in self.servers.keys()}

        for (s, servers) in self.servers.items():
            print 'Computing CCAs for subject', s
            T = self.num_periods[s] * self.num_subperiods
            cca_s = self.cca[s]
            tls_s = tls[s]

            for sp in xrange(T):
                sp_within_p = sp % self.num_subperiods
                subperiods = [ds.get_data().T for ds in servers]

                for (v, data) in enumerate(subperiods):
                    
                    if tls_s[v] is None:
                        tls_s[v][sp_within_p] = subperiods[v]
                    else:
                        tls_s[v][sp_within_p] = np.vstack([
                            tls_s[v], subperiods[v]])

        for (s, views) in tls.items():
            for v1 in xrange(self.num_views):
                for v2 in xrange(v1+1, self.num_views):
                    for sp in xrange(self.num_subperiods):
                        step1 = self.cols[v1]
                        step2 = self.cols[v2]
                        v1_mat = views[v1][sp]
                        v2_mat = views[v2][sp]
                        (v1_mat, v2_mat) = get_matched_dims(
                            v1_mat, v2_mat)

                        for f1 in xrange(step1):
                            v1_mat_f1 = v1_mat[f1::step1,:]

                            for f2 in xrange(step2):
                                v2_mat_f2 = v2_mat[f2::step2,:]
                                # TODO: only do sparse CCA if dimensionally necessary
                                ntpt = get_cca_vecs(
                                    v1_mat_f1, v2_mat_f2, num_nonzero=1)

                                self._save(
                                    ntpt,
                                    s,
                                    v1,
                                    v2,
                                    f1,
                                    f2,
                                    sp)

    def _save(self, ntpt, s, v1, v2, f1, f2, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_str = str(v1) + '-' + str(v2)
        
        if v_str not in s_group:
            s_group.create_group(v_str)

        v_group = s_group[v_str]
        f_str = str(i) + '-' + str(j)

        if f_str not in v_group:
            v_group.create_group(f_str)

        f_group = v_group[f_str]
        sp_str = str(sp)

        if sp_str not in f_group:
            f_group.create_group(sp_str)

        sp_group = v_group[sp_str]

        sp_group.create_dataset('Phi1', data=ntpt[0])
        sp_group.create_dataset('Phi2', data=ntpt[1])
        sp_group.create_dataset('CC', data=ntpt[2])

    def _load(self):

        for (s, spud) in self.cca.items():
            for (v1, v2) in spud.keys():
                l = {(f1, f2) : [None] * self.num_subperiods
                     for f1 in xrange(self.cols[v1])
                     for f2 in xrange(self.cols[v2])}

                spud.insert(v1, v2, l)
        
        for (s, s_group) in self.hdf5_repo.items():
            cca_s = self.cca[s]

            for (v_str, v_group) in s_group.items():
                (v1, v2) = [int(v) for v in v_str.split('-')]
                cca_vs = cca_s.get(v1, v2)

                for f_str in v_group.items():
                    f_tuple = tuple([int(f) for f in f_str.split('-')])
                    cca_fvs = cca_vs[f_tuple]

                    for (sp_str, sp_group) in v_group.items():
                        sp = int(sp_str)
                        ntpt = (
                                np.array(sp_group['Phi1']),
                                np.array(sp_group['Phi2']))
                        ntptcc = np.array(sp_group['CC'])
                        
                        cca_fvs[sp] = (ntpt, ntptcc)

    def _show(self):

        for (s, spud) in self.cca.items():  
            print 'Generating n_time_p_time plots for subject', s

            for ((v1, v2), freq_pairs) in spud.items():
                print '\tGenerating plots for view pair', v1, v2

                for ((f1, f2), subperiods) in freq_pairs.items():
                    print '\t\tGenerating plot frequency pair', f1, f2

                    fig = plt.figure()
                    (ntpt, ntptcc) = unzip(subperiods)
                    (Phi1s, Phi2s) = unzip(ntpt)
                    title = 'View-pairwise cca (n time p time) for ' + \
                        ' dim. ' + str(f1) + ' of view ' + self.names[v1] + \
                        ' dim. ' + str(f2) + ' of view ' + self.names[v2] + \
                        ' of subject ' + s
                    x_name = 'subperiod'
                    y_name = 'sample'
                    v_name = 'canonical vector value'

                    ax1 = fig.add_subplot(211)

                    self._plot_matrix_heat(
                        s,
                        v1,
                        Phi1s,
                        'subperiod',
                        'sample',
                        'canonical vector value',
                        ax1)

                    ax2 = fig.add_subplot(212)

                    self._plot_matrix_heat(
                        s,
                        v2,
                        Phi2s,
                        'subperiod',
                        'sample',
                        'canonical vector value',
                        ax2)

                    fig.suptitle(title)
                    fig.autofmt_xdate()

                    fn = '_'.join(title.split()) + '.png'
                    path = os.path.join(
                        self.plot_dir, fn)

                    fig.savefig(path, format='png')
                    plt.clf()

    def _plot_matrix_heat(self, s, v, ccal, x_name, y_name, v_name, ax):

        tl = np.hstack(ccal)
        (n, m) = tl.shape
        (yl, xl) = (np.arange(n), np.arange(m))

        # If canonical parameters are small, ceiling/floor to emphasize larger values
        if np.all(np.abs(tl) < 0.5):
            tl = np.copy(tl)
            threshold = 0.2 * np.max(np.abs(tl))
            tl_gt = tl > threshold
            tl_lt = tl < -threshold
            tl_middle = np.logical_not(
                np.logical_or(tl_gt, tl_lt))

            tl[tl_gt] = 1
            tl[tl_lt] = -1
            tl[tl_middle] = 0

        plot_matrix_heat(
            tl,
            xl,
            yl,
            '',
            x_name,
            y_name,
            v_name,
            vmax=1,
            vmin=-1,
            ax=ax)
