import os

import numpy as np

from math import log

from data.smart_watch import get_data_summaries
from plotting import plot_matrix_heat
from plotting.utils import get_plot_path

from bokeh.plotting import figure, show, output_file, vplot

def plot_label_counts(data_dir):

    labels = get_data_summaries(data_dir)['labels']
    counts = {}

    for label in labels:
        if label not in counts:
            counts[label] = 1 
        else:
            counts[label] += 1

    factors = counts.keys()
    x = counts.values()
    x_max = max(x)
    scale = 10**(log(x_max, base=10))
    x_max = (x_max - x_max % scale) + scale
    p = figure(title="Label Counts", tools="resize,save", y_range=factors, x_range=[0,x_max])

    p.segment(0, factors, x, factors, line_width=2, line_color="green", )
    p.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

    filepath = os.path.join(get_plot_path(), "label_counts.html")
    
    output_file(filepath, title="label counts")
    show(p)

def plot_canonical_bases(Phi, Psi):

    (p1, k1) = Phi.shape
    (p2, k2) = Psi.shape
    k = None
    
    if k1 == k2:
        k = k1
    else:
        raise ValueError(
            'Second dimension of each basis should be equal.')

    filepath = os.path.join(
        get_plot_path(), 
        'mass_per_feature_over_bases_matrix_heat_plot.html')

    Phi_features = [str(i) for i in range(p1)]
    Psi_features = [str(i) for i in range(p2)]
    basis_elements = [str(i) for i in range(k)]

    Phi_p = _plot_basis(Phi, 'Phi', Phi_features, basis_elements)
    Psi_p = _plot_basis(Psi, 'Psi', Psi_features, basis_elements)
    
    output_file(filepath, 'percent mass per feature over bases')
    show(vplot(Phi_p, Psi_p))

def _plot_basis(basis, name, features, basis_elements):

    return plot_matrix_heat(
        np.abs(basis),
        basis_elements,
        features,
        'Percent mass per feature over ' + name + ' basis elements',
        name + ' basis element',
        'feature',
        'mass')
