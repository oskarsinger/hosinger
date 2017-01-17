import os
import matplotlib
import h5py

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.arithmetic import get_running_avg
from drrobert.file_io import get_timestamped as get_ts
from drrobert.stats import get_pearson_matrix as get_pm
from lazyprojector import plot_matrix_heat
from data.loaders.e4.utils import get_symptom_status
from data.loaders.readers.from_num import get_array_as_is as get_aai

class ViewPairwiseCorrelationRunner:

    def __init__(self,
        dtcwt_runner,
        save_load_dir,
	wavelets=None,
        save=False,
        load=False,
        show=False,
	avg_over_subjects=False):

        self.save = save
        self.load = load
        self.show = show
	self.avg_over_subjects = avg_over_subjects

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.wavelets = dtcwt_runner.wavelets \
		if wavelets is None else wavelets
	self.subjects = self.wavelets.keys()
        self.names = dtcwt_runner.names
        self.names2indices = {name : i 
                              for (i, name) in enumerate(self.names)}
        self.num_views = dtcwt_runner.num_views
        self.num_periods = {s : len(self.wavelets[s])
			    for s in self.subjects}
	self.max_periods = max(self.num_periods.values())

	if self.avg_over_subjects:
	    self.subjects = {s for (s, np) in self.num_periods.items()
			     if np == self.max_periods}

        self.num_subperiods = dtcwt_runner.num_sps

        default = lambda: [[] for i in xrange(self.num_subperiods)]

        self.correlation = {s : SPUD(self.num_views, default=default)
                            for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

        if self.show:
            self._show_corr_over_time()

            self._show_corr_over_periods()
            self._show_corr_over_subperiods()

            self._show_corr_mean_over_periods()
            self._show_corr_mean_over_subperiods()

    def _init_dirs(self, 
        save, 
        load, 
        show, 
        save_load_dir):

        if (show or save) and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        hdf5_path = os.path.join(
            self.save_load_dir,
            'correlation.hdf5')
        self.hdf5_repo = h5py.File(
            hdf5_path, 'w' if save else 'r')
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir) 

    def _compute(self):

        for (s, s_wavelets) in self.wavelets.items():
            print 'Computing correlations for subject', s
            spud = self.correlation[s]

            for (p, day) in enumerate(s_wavelets):
                for (sp, subperiod) in enumerate(day):
                    for k in spud.keys():
                        (Yh1, Yl1) =  subperiod[k[0]]
                        (Yh2, Yl2) =  subperiod[k[1]]
                        Y1_mat = rmu.get_padded_wavelets(Yh1, Yl1)
                        Y2_mat = rmu.get_padded_wavelets(Yh2, Yl2)
                        (n1, p1) = Y1_mat.shape
                        (n2, p2) = Y2_mat.shape

                        if n1 < n2:
                            num_reps = int(float(n2) / n1)
                            repped = np.zeros((n2, p1), dtype=complex)
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    Y1_mat[:max_len,:])

                            Y1_mat = repped

                        elif n2 < n1:
                            num_reps = int(float(n1) / n2)
                            repped = np.zeros((n1, p2), dtype=complex)
                            
                            for r in xrange(num_reps):
                                max_len = repped[r::num_reps,:].shape[0]
                                repped[r::num_reps,:] = np.copy(
                                    Y2_mat[:max_len,:])

                            Y2_mat = repped
                        
                        correlation = get_pm(
                            np.absolute(Y1_mat), 
                            np.absolute(Y2_mat))

                        self._save(
                            correlation,
                            s,
                            k,
                            p,
                            sp)
                     
    def _save(self, c, s, v, p, sp):

        if s not in self.hdf5_repo:
            self.hdf5_repo.create_group(s)

        s_group = self.hdf5_repo[s]
        v_string = str(v[0]) + '-' + str(v[1])

        if v_string not in s_group:
            s_group.create_group(v_string)

        v_group = s_group[v_string]
        sp_string = str(sp)

        if sp_string not in v_group:
            v_group.create_group(sp_string)

        sp_group = v_group[sp_string]
        p_string = str(p)

        sp_group.create_dataset(p_string, data=c)

    def _load(self):

        for (s, spud) in self.correlation.items():
            for (k, subperiods) in spud.items():
                for sp in xrange(self.num_subperiods):
                    subperiods[sp] = [None] * self.num_periods[s] 
                
        for s in self.subjects:
            s_group = self.hdf5_repo[s]
            print 'Loading corr for subject', s

            for (k_str, sp_group) in s_group.items():
                vs = [int(v) for v in k_str.split('-')]

                for (sp_str, p_group) in sp_group.items():
                    sp = int(sp_str)

                    for (p_str, corr) in p_group.items():
                        p = int(p_str)
                        
            	        self.correlation[s].get(vs[0], vs[1])[sp][p] = get_aai(corr)

    def _show_corr_over_time(self):

        print 'Plotting correlation over time'

        avgs = SPUD(
            self.num_views, default=lambda: {})
        get_normed = lambda p: float(p) / float(self.num_subperiods)
	sample = self.correlation.values()[0].items()
	y_labels = SPUD(self.num_views)

	for (k, subperiods) in sample:
            (n, m) = subperiods[0][0].shape
	    y_labels_k = [rmu.get_2_digit_pair(i,j)
                     	  for i in xrange(n)
                          for j in xrange(m)]
            y_labels.insert(
		k[0], k[1], y_labels_k)

        for s in self.subjects:
	    spud = self.correlation[s]
	    sympt = get_symptom_status(s)
            num_points = self.num_periods[s] * self.num_subperiods
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(self.num_views, default=default)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                timeline = np.hstack(
                    [rmu.get_ravel_hstack(corrs) for corrs in periods])

                if self.avg_over_subjects:
                    current = avgs.get(k[0], k[1])

                    if sympt in current:
                        current[sympt] += timeline
                    else:
                        current[sympt] = timeline
                else:
		    (name1, name2) = (self.names[k[0]], self.names[k[1]])
                    title = 'View-pairwise correlation over' + \
                        ' time for views' + \
                        name1 + ' ' + name2 + \
                        ' of subject ' + s
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)
                    y_labels_k = y_labels.get(k[0], k[1])
                    x_labels = ['{:06.3f}'.format(get_normed(p))
                                for p in xrange(num_points)]

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels_k,
                        title,
                        'period',
                        'frequency pair',
                        'correlation',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()

	if self.avg_over_subjects:
            num_points = self.num_subperiods * self.max_periods

	    for (k, sympts) in avgs.items():
                y_labels_k = y_labels.get(k[0], k[1])
                x_labels = ['{:06.3f}'.format(get_normed(p))
                            for p in xrange(num_points)]
		(name1, name2) = (self.names[k[0]], self.names[k[1]])

                for (sympt, timeline) in sympts.items():
                    title = 'View-pairwise correlation over' + \
                        ' time for views ' + \
                        name1 + ' ' + name2 + \
                        ' with symptom status ' + sympt
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels_k,
                        title,
                        'time',
                        'frequency pair',
                        'correlation',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()

    def _show_corr_mean_over_subperiods(self):

        print 'Plotting correlation mean over subperiods'

        for (s, spud) in self.correlation.items():
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(self.num_views, default=default)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                (n, m) = periods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_periods[s])]
                timelines = [rmu.get_ravel_hstack(subperiods)
                             for subperiods in periods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])

		if self.avg_over_subjects:
		    print 'Poop'
		else:
                    title = 'View-pairwise mean-over-hours' + \
                        ' correlation over days for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                    	' of subject ' + s
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                    	timeline,
			x_labels,
			y_labels,
			title,
			'day',
			'frequency pair',
			'correlation',
			 vmax=1,
			 vmin=-1)[0].get_figure().savefig(
                             path, format='pdf')
		    sns.plt.clf()

	if self.avg_over_subjects:
	    print 'Poop'

    def _show_corr_over_subperiods(self):

        print 'Plotting correlation over subperiods'

        for (s, spud) in self.correlation.items():
            default = lambda: [[] for p in xrange(self.num_periods[s])]
            period_corrs = SPUD(self.num_views, default=default)

            for (k, subperiods) in spud.items():
                for periods in subperiods:
                    for (p, corr) in enumerate(periods):
                        period_corrs.get(k[0], k[1])[p].append(corr)

            for (k, periods) in period_corrs.items():
                (n, m) = periods[0][0].shape
                y_labels = [rmu.get_2_digit_pair(i,j)
                            for i in xrange(n)
                            for j in xrange(m)]
                x_labels = [rmu.get_2_digit(sp, power=False)
                            for sp in xrange(self.num_subperiods)]
                name1 = self.names[k[0]]
                name2 = self.names[k[1]]

                for (p, subperiods) in enumerate(periods):
                    timeline = rmu.get_ravel_hstack(subperiods)

		    if self.avg_over_subjects:
			print 'Poop'
		    else:
		        title = 'View-pairwise correlation over ' + \
			    ' hours for views ' + name1 + ' ' + \
			    name2 + ' of subject ' + s + \
			    ' and day ' + rmu.get_2_digit(p)
		        fn = '_'.join(title.split()) + '.pdf'
		        path = os.path.join(self.plot_dir, fn)

		        plot_matrix_heat(
			    timeline,
			    x_labels,
			    y_labels,
			    title,
			    'hour',
			    'frequency pair',
			    'correlation',
			    vmax=1,
			    vmin=-1)[0].get_figure().savefig(
			        path, format='pdf')
		        sns.plt.clf()

	if self.avg_over_subjects:
	    print 'Poop'

    def _show_corr_mean_over_periods(self):

        print 'Plotting correlation mean over periods'

	avgs = SPUD(self.num_views, default=lambda: {})
	things = self.correlation.values()[0].items()
	y_labels = SPUD(self.num_views)

	for (k, subperiods) in things:
            (n, m) = subperiods[0][0].shape
	    y_labels_k = [rmu.get_2_digit_pair(i,j)
                     	  for i in xrange(n)
                          for j in xrange(m)]
            y_labels.insert(
		k[0], k[1], y_labels_k)

        for s in self.subjects:
	    spud = self.correlation[s]
	    sympt = get_symptom_status(s)

            print 'Producing timelines for subject', s

            for (k, subperiods) in spud.items():
                (n, m) = subperiods[0][0].shape
                y_labels_k = y_labels.get(k[0], k[1])
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_subperiods)]
                timelines = [rmu.get_ravel_hstack(periods)
                             for periods in subperiods]
                timeline = np.hstack(
                    [np.mean(tl, axis=1)[:,np.newaxis] 
                     for tl in timelines])

		if self.avg_over_subjects:
		    current = avgs.get(k[0], k[1])

		    if sympt in current:
			current[sympt] += timeline
		    else:
			current[sympt] = timeline
		else:
                    title = 'View-pairwise mean-over-periods' + \
                        ' correlation over subperiods for views ' + \
                        self.names[k[0]] + ' ' + self.names[k[1]] + \
                        ' of subject ' + s
                    fn = '_'.join(title.split()) + '.pdf'
                    path = os.path.join(self.plot_dir, fn)

                    plot_matrix_heat(
                        timeline,
                        x_labels,
                        y_labels,
                        title,
                        'subperiod',
                        'frequency pair',
                        'correlation',
                        vmax=1,
                        vmin=-1)[0].get_figure().savefig(
                            path, format='pdf')
                    sns.plt.clf()

	if self.avg_over_subjects:
	    for (k, sympts) in avgs.items():
                print 'Plotting for views', self.names[k[0]], self.names[k[1]]
                y_labels_k = y_labels.get(k[0], k[1])
                x_labels = ['{:02f}'.format(p)
                            for p in xrange(self.num_subperiods)]
		(name1, name2) = (self.names[k[0]], self.names[k[1]])

		for (sympt, timeline) in sympts.items():
                    title = 'View-pairwise mean-over-periods' + \
                        ' correlation over subperiods for views ' + \
                        name1 + ' ' + name2 + \
			' with symptom status ' + sympt
		    fn = '_'.join(title.split()) + '.pdf'
		    path = os.path.join(self.plot_dir, fn)

		    plot_matrix_heat(
			timeline,
			x_labels,
			y_labels_k,
			title,
			'subperiod',
			'frequency pair',
			'correlation',
			vmax=1,
			vmin=-1)[0].get_figure().savefig(
			    path, format='pdf')
		    sns.plt.clf()

    def _show_corr_over_periods(self):

        print 'Plotting correlation over periods'

	default = lambda: [{} for i in xrange(self.num_subperiods)]
	avgs = SPUD(self.num_views, default=default)
	things = self.correlation.values()[0].items()
	y_labels = SPUD(self.num_views)

	for (k, subperiods) in things:
            (n, m) = subperiods[0][0].shape
	    y_labels_k = [rmu.get_2_digit_pair(i,j)
                     	  for i in xrange(n)
                          for j in xrange(m)]
            y_labels.insert(
		k[0], k[1], y_labels_k)

        for s in self.subjects:
	    spud = self.correlation[s]
	    sympt = get_symptom_status(s)

            for (k, subperiods) in spud.items():
                y_labels_k = y_labels.get(k[0], k[1])
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.num_periods[s])]
		(name1, name2) = (self.names[k[0]], self.names[k[1]])

                for (sp, periods) in enumerate(subperiods):
                    timeline = rmu.get_ravel_hstack(periods)

		    if self.avg_over_subjects:
			current = avgs.get(k[0], k[1])[sp]

			if sympt in current:
			    current[sympt] += timeline
			else:
			    current[sympt] = timeline
		    else:
                    	title = 'View-pairwise correlation over' + \
			    ' periods for views' + \
                            name1 + ' ' + name2 + \
                            ' of subject ' + s + \
			    ' at subperiod ' + str(sp)
                        fn = '_'.join(title.split()) + '.pdf'
                        path = os.path.join(self.plot_dir, fn)

                        plot_matrix_heat(
			    timeline,
			    x_labels,
			    y_labels_k,
			    title,
			    'period',
			    'frequency pair',
			    'correlation',
			    vmax=1,
			    vmin=-1)[0].get_figure().savefig(
                                path, format='pdf')
                        sns.plt.clf()

	if self.avg_over_subjects:
	    for (k, subperiods) in avgs.items():
                y_labels_k = y_labels.get(k[0], k[1])
                x_labels = [rmu.get_2_digit(p, power=False)
                            for p in xrange(self.max_periods)]
		(name1, name2) = (self.names[k[0]], self.names[k[1]])

		for (sp, sympts) in enumerate(subperiods):
		    for (sympt, timeline) in sympts.items():
                        title = 'View-pairwise correlation over' + \
			    ' periods for views ' + \
                            name1 + ' ' + name2 + \
			    ' at subperiod ' + str(sp) + \
			    ' with symptom status ' + sympt
		        fn = '_'.join(title.split()) + '.pdf'
		        path = os.path.join(self.plot_dir, fn)

		        plot_matrix_heat(
			    timeline,
			    x_labels,
			    y_labels_k,
			    title,
			    'period',
			    'frequency pair',
			    'correlation',
			    vmax=1,
			    vmin=-1)[0].get_figure().savefig(
			        path, format='pdf')
		        sns.plt.clf()
