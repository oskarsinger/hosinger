import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_plot_path

def plot_lines(
    data_map,
    x_label,
    y_label,
    title,
    colors=Spectral11):

    p = figure(
        title=title,
        plot_width=width, plot_height=height,
        tools="resize,hover,save,box_zoom")
    p.grid.grid_line_alpha=0.3
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label

    count = 0

    for name, (x_data, y_data) in data_map.items():
        p.line(
            x_data, 
            y_data, 
            color=colors[count % len(colors)], 
            legend=name)

        count += 1

    return p

def _get_dataframe(data_map):

    names = []
    xs = None
    ys = None
    units = []

    for name, (x_data, y_data) in data_map.items():
        names.extend(
            [name for i in xrange(x_data.shape[0])])

        if xs is None:
            xs = x_data
        else:
            xs = np.vstack([xs, x_data])

        if ys is None:
            ys = y_data
        else:
            ys = np.vstack([ys, y_data])

        units = [1] * x_data.shape[0]

    # TODO: finish putting this into a dataframe
