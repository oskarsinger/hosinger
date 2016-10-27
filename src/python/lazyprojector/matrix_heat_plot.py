import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import linal.utils.misc as lum
import matplotlib.pyplot as plt

from math import pi
from utils import get_plot_path

def plot_matrix_heat(
    value_matrix, 
    x_labels, 
    y_labels, 
    title,
    x_name,
    y_name,
    val_name,
    do_phase=False):

    (n, m) = value_matrix.shape
    value_matrices = None
    ps = []

    if np.any(np.iscomplex(value_matrix)):
        value_matrices = {
            'magnitude': np.absolute(value_matrix)}

        if do_phase:
            value_matrices['phase'] = np.angle(value_matrix)
    else:
        value_matrices = {
            'magnitude': value_matrix}

    plots = []

    for k, matrix in value_matrices.items():
        source = _populate_data_source(
            matrix, 
            x_labels, 
            y_labels,
            x_name,
            y_name,
            val_name)
        ax = plt.axes()
        plot = sns.heatmap(
            source,
            yticklabels=4,
            ax=ax)

        ax.set_title(title + ' ' + k)

        for label in plot.get_yticklabels():
            label.set_rotation(45)

        plots.append(plot)
        
    return plots

def _populate_data_source(
    value_matrix, 
    x_labels, 
    y_labels,
    x_name,
    y_name,
    val_name):

    (n, p) = value_matrix.shape
    x_element = []
    y_element = []
    values = []

    for j in xrange(p):
        for i in xrange(n):
            x_element.append(x_labels[j])
            y_element.append(y_labels[i])
            values.append(value_matrix[i,j])

            d = {
                x_name: x_element,
                y_name: y_element,
                val_name: values}
            df = pd.DataFrame(data=d).pivot(
                y_name,
                x_name,
                val_name)

    return df
