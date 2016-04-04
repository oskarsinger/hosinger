import os

import numpy as np

from math import log

from data.smart_watch import get_data_summaries
from plotting import plot_matrix_heat, plot_lines
from plotting.utils import get_plot_path

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import GnBu9, YlOrRd9

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

    filepath = get_plot_path("label_counts")
    
    output_file(filepath, title="label counts")
    show(p)

def plot_cca_filtering(
    filtered_X,
    filtered_Y,
    X_width=900,
    X_height=400,
    Y_width=900,
    Y_hieght=400):

    (n1, k1) = filtered_X.shape
    (n2, k2) = filtered_Y.shape

    if (not n1 == n2) or (not k1 == k2):
        raise ValueError(
            'filtered_X and filtered_Y should have the same dimensions.')

    (n, k) = (n1, k1)
    X_map = {'X filter dimension ' + str(i) : filtered_X[:,i]
             for i in xrange(k)}
    Y_map = {'Y filter dimension ' + str(i) : filtered_Y[:,i]
             for i in xrange(k)}
    X_plot = plot_lines(
        X_map,
        "Time Step",
        "Filtered X Data Point",
        "Filtered X Data Points vs Time Step Observed",
        color=YlGnBu9,
        width=X_width,
        heigh=X_height)

    Y_plot = plot_lines(
        Y_map,
        "Time Step",
        "Filtered Y Data Point",
        "Filtered Y Data Points vs Time Step Observed",
        color=YlOrRd9,
        width=Y_width,
        height=Y_height)

    filepath = get_plot_path(
        'cca_filtered_data_points_vs_time_step_observed')

    output_file(
        filepath, 
        'CCA-filtered data points vs time step observed')
    show(vplot(X_plot, Y_plot))

def plot_canonical_bases(Phi, Psi):

    (p1, k1) = Phi.shape
    (p2, k2) = Psi.shape
    k = None
    
    if k1 == k2:
        k = k1
    else:
        raise ValueError(
            'Second dimension of each basis should be equal.')

    Phi_features = [str(i) for i in range(p1)]
    Psi_features = [str(i) for i in range(p2)]
    basis_elements = [str(i) for i in range(k)]

    Phi_p = _plot_basis(Phi, 'Phi', Phi_features, basis_elements)
    Psi_p = _plot_basis(Psi, 'Psi', Psi_features, basis_elements)
    
    filepath = get_plot_path(
        'mass_per_feature_over_bases_matrix_heat_plot')

    output_file(
        filepath, 
        'percent mass per feature over bases')
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
