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
    (x, y) = _get_labels(
        n, p,
        x_labels, 
        y_labels)
    plot = ax.pcolormesh(
        x, y, 
        value_matrix,
        cmap='RdBu',
        vmax=vmax,
        vmin=vmin)

    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(plot)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

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

def _get_labels(
    n, p,
    x_labels, 
    y_labels):

    x = np.zeros((n,p))
    y = np.zeros((n,p))

    for i in xrange(n):
        x[i,:] = np.copy(x_labels)
        
    for i in xrange(p):
        y[:,i] = np.copy(y_labels)

    return (x, y)
