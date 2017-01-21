import matplotlib

matplotlib.use('Cairo')

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
    vmax=None,
    vmin=None,
    ax=None):

    (n, m) = value_matrix.shape

    if ax is None:
        ax = plt.axes()

    (n, p) = value_matrix.shape
    source = _populate_data_source(
        value_matrix, 
        x_labels, 
        y_labels,
        x_name,
        y_name,
        val_name)
    plot = sns.heatmap(
        source,
        ax=ax,
        vmax=vmax,
        vmin=vmin)

    if n > 10:
        plot.yticklabels = n / 5

    if p > 10:
        plot.xticklabels = p / 5

    ax.set_title(title)

    for label in plot.get_yticklabels():
        label.set_rotation(45)

    for label in plot.get_xticklabels():
        label.set_rotation(45)

    return plot

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
