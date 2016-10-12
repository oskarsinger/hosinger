import os

import numpy as np
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from drrobert.ml import get_kmeans
from lazyprojector import plot_matrix_heat
from bokeh.models.layouts import Column, Row

class ViewPairwiseCorrelationRunner:

    def __init__(self, 
        wavelets,
        save_load_dir,
        save=False,
        load=False,
        show=False,
        k=None):

        self.wavelets = wavelets
        self.save = save
        self.load = load
        self.show = show
        self.k = k

        self._init_dirs(
            save, 
            load, 
            show, 
            save_load_dir)

        self.subjects = self.wavelets.subjects
        self.names = self.wavelets.names
        self.num_views = self.wavelets.num_views
        self.num_periods = self.wavelets.num_periods
        self.correlation = rmu.get_list_spud_dict(
            self.num_views,
            self.subjects,
            no_double=True)

    def run(self):

        if self.load:
            self._load()
        else:
            self._load_wavelets()
            self._compute()

        if self.show:
            self._show()

    def _init_dirs(self, save, load, show, save_load_dir):

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('VPWCR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.corr_dir = rmu.init_dir(
            'correlation',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir) 

    def _load(self):

        print 'Loading correlation'

        correlation = {}

        for fn in os.listdir(self.corr_dir):
            path = os.path.join(self.corr_dir, fn)

            with open(path) as f:
                correlation[fn] = np.load(f)

        for (s, spud) in self.correlation.items():
            for k in spud.keys():
                l = [None] * self.num_periods[subject]

                spud.insert(k[0], k[1], l)

        for (k, hm) in correlation.items():
            info = k.split('_')
            s = info[1]
            p = int(info[3])
            vs = [int(i) for i in info[5].split('-')]
            
            self.correlation[s].get(vs[0], vs[1])[p] = hm

    def _compute(self):

        for subject in self.subjects:

            print 'Computing correlation for subject', subject

            for (period, (Yhs, Yls)) in enumerate(self.wavelets[subject]):
                correlation = self._get_period(Yhs, Yls)

                for ((i, j), corr) in correlation.items():
                    self.correlation[subject].get(i, j).append(corr)

                if self.save:
                    for (k, hm) in correlation.items():
                        path = '_'.join([
                            'subject',
                            subject,
                            'period',
                            str(period),
                            'views',
                            '-'.join([str(j) for j in k]),
                            'dtcwt_correlation_matrix.thang'])
                        path = os.path.join(self.corr_dir, path)

                        with open(path, 'w') as f:
                            np.save(f, hm)

    def _get_period(self, Yhs, Yls):

        Yh_matrices = [rmu.get_sampled_wavelets(Yh, Yl)
                       for (Yh, Yl) in zip(Yhs, Yls)]
        min_length = min(
            [Y.shape[0] for Y in Yh_matrices]) 
        rates = [int(Y.shape[0] / min_length)
                 for Y in Yh_matrices]
        subsamples = [m[::r,:]
                      for (m, r) in zip(Yh_matrices, rates)]
        get_matrix = lambda i,j: rmu.get_normed_correlation(
            subsamples[i].T, subsamples[j])
        correlation = SPUD(self.num_views)

        for (i, j) in correlation.keys():
            mat = get_matrix(i, j)
            correlation.insert(i, j, mat)

        return correlation

    def _show(self):

        timelines = SPUD(self.num_views, default=list)

        for subject in self.subjects:

            print 'Producing correlation plots for subject', subject

            for (k, l) in self.correlation[subject].items():
                self._plot_correlation(k, l, subject)

    def _plot_correlation(self, key, timeline, subject):

        (i, j) = key
        (n, p) = timeline[0].shape
        title = 'Correlation of views ' + \
            self.names[i] + ' and ' + self.names[j] + \
            ' by decimation level' + \
            ' for subject ' + subject
        x_name = 'decimation level'
        y_name = 'decimation level'
        x_labels = ['2^' + str(-k) for k in xrange(p)]
        y_labels = ['2^' + str(-k) for k in xrange(n)]
        val_name = 'correlation'
        plots = []

        for (l, hm) in enumerate(timeline):
            hmp = plot_matrix_heat(
                hm,
                x_labels,
                y_labels,
                title,
                x_name,
                y_name,
                val_name,
                width=p*50,
                height=n*50)

            plots.append(hmp)

        plot = Column(*plots)
        filename = '_'.join([
            'correlation_of_wavelet_coefficients',
            'subject',
            subject,
            self.names[i], 
            self.names[j]]) + '.html'
        filepath = os.path.join(self.plot_dir, filename)

        output_file(
            filepath, 
            'correlation_of_wavelet_coefficients_' +
            self.names[i] + '_' + self.names[j])
        show(plot)
        
    def _show_corr_subblocks(self, begin_end_spud):

        plots = {s : [] for s in self.subjects}

        for (s, spud) in self.correlation.items():
            for ((i, j), corr_list) in spud.items():
                (by, ey, bx, ex) = begin_end_spud.get(i, j)

                if ey is None:
                    ey = corr_list[0].shape[0]

                if ex is None:
                    ex = corr_list[0].shape[1]

                subblocks = [np.ravel(
                                corr[by:ey,bx:ex])[:,np.newaxis]
                             for corr in corr_list]
                matrix = np.hstack(subblocks)
                get_label = lambda k,l: ' '.join([
                    self.names[i],
                    '2^' + str(k),
                    'vs',
                    self.names[j],
                    '2^' + str(j)])
                y_labels = [get_label(k, l)
                            for k in xrange(by, ey)
                            for l in xrange(bx, ex)]
                x_labels = [str(k) 
                            for k in xrange(self.num_periods[s])]
                title = ' '.join([
                    'High mag. corr. for freq. pair',
                    self.names[i], self.names[j],
                    'vs time for subject', s])
                x_name = 'Time'
                y_name = 'Frequency pairs'
                val_name = 'Correlation'
                plot = plot_matrix_heat(
                    matrix,
                    x_labels,
                    y_labels,
                    title,
                    x_name,
                    y_name,
                    val_name,
                    width=150*matrix.shape[1],
                    height=50*matrix.shape[0],
                    norm_axis=0,
                    do_phase=self.do_phase)

                plots[s].append(plot)
            
        col_plots = {s: Column(*ps)
                     for (s, ps) in plots.items()}

        for (s, cp) in col_plots.items():
            filename = '_'.join([
                'high_mag_correlation_of_wavelet_coefficient_magnitude',
                'subject', s]) + '.html'
            filepath = os.path.join(self.plot_dir, filename)

            output_file(
                filepath, 
                'high_mag_correlation_of_wavelet_coefficient_magnitude')
            show(plot)

    def _compute_correlation_kmeans(self):

        print 'Computing correlation k-means'

        data = SPUD(self.num_views, default=list)
        subjects = SPUD(self.num_views, default=list)
        
        for subject in self.subjects:
            items = self.correlation[subject].items()
            
            for ((i, j), corr_list) in items:
                raveled_corrs = [np.ravel(corr) for corr in corr_list]

                data.get(i, j).extend(
                    [np.ravel(corr) for corr in corr_list])
                subjects.get(i, j).extend(
                    [subject] * len(corr_list))

        mag = SPUD(self.num_views)

        for ((i, j), v) in data.items():
            corrs_as_rows = np.vstack(v)

            mag.insert(i, j, np.absolute(corrs_as_rows))

        self.kmeans = rmu.get_kmeans_spud_dict(
            mag, subjects, self.k, self.num_views, self.subjects)
