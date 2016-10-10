import os

import numpy as np
import utils as rmu

from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from drrobert.file_io import get_timestamped as get_ts
from lazyprojector import plot_matrix_heat
from bokeh.palettes import BuPu9, Oranges9
from bokeh.models.layouts import Column, Row
from bokeh.plotting import output_file, show
from sklearn.cross_decomposition import CCA

class ViewPairwiseCCARunner:

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

        self.names = self.wavelets.names
        self.num_views = wavelets.num_views
        self.subjects = wavelets.subjects
        self.num_periods = wavelets.num_periods
        self.cca = rmu.get_list_spud_dict(
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

            model_dir = get_ts('VPWCCAR')

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.cca_dir = rmu.init_dir(
            'cca',
            save,
            self.save_load_dir)
        self.plot_dir = rmu.init_dir(
            'plots',
            show,
            self.save_load_dir) 

    def _load(self):

        print 'Loading CCA'

        cca = {}

        for fn in os.listdir(self.cca_dir):
            path = os.path.join(self.cca_dir, fn)

            with open(path) as f:
                cca[fn] = np.load(f)

        keys = self.cca[self.subjects[0]].keys()

        for subject in self.subjects:
            num_periods = self.num_periods[subject]

            for (i, j) in keys:
                self.cca[subject].insert(
                    i, j, [{} for k in xrange(num_periods)])

        for (k, mat) in cca.items():
            info = k.split('_')
            name = info[0]
            subject = info[2]
            period = int(info[4])
            views = [int(i) for i in info[6].split('-')]

            self.cca[subject].get(
                views[0], views[1])[period][name] = mat
    
    def _compute(self):

        for subject in self.subjects:

            print 'Computing CCA for subject', subject

            cca = rmu.get_list_spud_dict(
                self.num_views,
                self.subjects,
                no_double=True)

            for (period, (Yhs, Yls)) in enumerate(self.wavelets[subject]):

                print 'Computing CCA for period', period

                #TODO: what are all the different 'current's here for?
                wavelet_matrices = [rmu.get_sampled_wavelets(Yh, Yl)
                                    for (Yh, Yl) in zip(Yhs, Yls)]
                wms = [np.absolute(wm) for wm in wavelet_matrices]
                current = _get_cca_spud(wms)

                cca[subject] = _get_appended_spud(
                    cca[subject], current)

                if self.save_cca:
                    self._save_cca(subject, current, period)

        self.cca = cca

    def _save_cca(self, subject, current, period):

        for (k, xy_pair) in current.items():
            path = '_'.join([
                'subject',
                subject,
                'period',
                str(period),
                'views',
                '-'.join([str(j) for j in k]),
                'dtcwt_cca_matrix.thang'])

            for (l, mat) in xy_pair.items():
                if self.cca_dir is not None:
                    current_path = os.path.join(
                        self.cca_dir, l + '_' + path)

                with open(current_path, 'w') as f:
                    np.save(f, mat)

    def _show(self):

        for subject in self.subjects:

            print 'Producing CCA plots for subject', subject

            for (k, l) in self.cca[subject].items():
                self._plot_cca(k, l, subject)

    def _plot_cca(self, key, timeline, subject):

        (i, j) = key
        (nx, px) = timeline[0]['Xw'].shape
        (ny, py) = timeline[0]['Yw'].shape
        title = 'CCA decomposition of magnitude' + \
            ' of views ' + \
            self.names[i] + ' and ' + self.names[j] + \
            ' by decimation level' + \
            ' for subject ' + subject
        x_title = 'CCA transform for view' + self.names[i]
        y_title = 'CCA transform for view' + self.names[j]
        x_name = 'period'
        y_name = 'decimation level'
        x_labels = [str(k) for k in xrange(len(timeline))]
        yx_labels = ['2^' + str(-k) for k in xrange(nx)]
        yy_labels = ['2^' + str(-k) for k in xrange(ny)]
        val_name = 'cca parameter'
        plots = []

        X_t = np.hstack(
            [t['Xw'] for t in timeline])
        Y_t = np.hstack(
            [t['Yw'] for t in timeline])

        X_pos_color_scheme = list(reversed(BuPu9))
        X_neg_color_scheme = list(reversed(Oranges9))
        X_plot = plot_matrix_heat(
            X_t,
            x_labels,
            yx_labels,
            title,
            x_name,
            y_name,
            val_name,
            width=150*X_t.shape[1],
            height=50*X_t.shape[0],
            pos_color_scheme=X_pos_color_scheme,
            neg_color_scheme=X_neg_color_scheme,
            norm_axis=0,
            do_phase=self.do_phase)
        Y_plot = plot_matrix_heat(
            Y_t,
            x_labels,
            yy_labels,
            title,
            x_name,
            y_name,
            val_name,
            width=150*X_t.shape[1],
            height=50*X_t.shape[0],
            norm_axis=0,
            do_phase=self.do_phase)

        plot = Column(*[X_plot, Y_plot])
        filename = '_'.join([
            'cca_of_wavelet_coefficients',
            'subject',
            subject,
            phase_or_mag,
            self.names[i], 
            self.names[j]]) + '.html'
        filepath = os.path.join(self.plot_dir, filename)

        output_file(
            filepath, 
            'cca_of_wavelet_coefficients_' +
            self.names[i] + '_' + self.names[j])
        show(plot)

    def _compute_kmeans(self, pw_cca):
        data = SPUD(
            self.num_views, 
            default=list, 
            no_double=True)

        subjects = SPUD(
            self.num_views, 
            default=list, 
            no_double=True)

        for subject in self.subjects:
            s_pw_cca = pw_cca[subject]

            for ((i, j), pairs) in s_pw_cca.items():
                stack = lambda p: np.ravel(
                    np.vstack([p['Xw'], p['Yw']]))
                l = [stack(p) for p in pairs]

                data.get(i, j).extend(l)
                subjects.get(i, j).extend([subject] * len(l))

        matrices = SPUD(self.num_views, no_double=True)

        for ((i, j), v) in data.items():
            as_rows = np.vstack(v)

            matrices.insert(i, j, as_rows)

        self.kmeans = rmu.get_kmeans_spud_dict(
            matrices, 
            subjects, 
            self.k, 
            self.num_views, 
            self.subjects)

def _get_cca_spud(views):

    num_views = len(views)
    current = SPUD(num_views, no_double=True)

    for i in xrange(num_views):
        for j in xrange(i+1, num_views):
            X_data = views[i]
            Y_data = views[j]
            print X_data, Y_data
            cca = CCA(n_components=1)

            cca.fit(X_data, Y_data)

            xy_pair = {
                'Xw': cca.x_weights_,
                'Yw': cca.y_weights_}

            current.insert(i, j, xy_pair)

    return current
