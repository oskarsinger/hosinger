import numpy as np

from math import pi

from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, show, output_file
from bokeh.palettes import Spectral10

from utils import get_plot_path

def plot_matrix_heat(
    value_matrix,
    x_labels, 
    y_labels, 
    title,
    x_name,
    y_name,
    val_name,
    color_scheme=Spectral10
    norm_axis=0,
    width=900,
    height=400):

    if not np.all(value_matrix >= 0):
        raise ValueError(
            'Elements of value_matrix must all be non-negative.')

    source = _populate_data_source(
        value_matrix, 
        x_labels, 
        y_labels,
        norm_axis,
        color_scheme)
    p = _initialize_figure(
        source,
        width,
        height,
        title,
        x_labels,
        y_labels,
        x_name,
        y_name,
        val_name)

    return p

def _initialize_figure(
    source, 
    width,
    height,
    title,
    x_labels, 
    y_labels, 
    x_name, 
    y_name,
    val_name):

    TOOLS = "resize,hover,save,pan,box_zoom,wheel_zoom"

    p = figure(
        title=title,
        x_range=x_labels, y_range=list(reversed(y_labels)),
        plot_width=width, plot_height=height,
        x_axis_location="above", toolbar_location="left", 
        tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi/3

    p.rect('x_element', 'y_element', 1, 1, source=source,
           color='color', line_color=None)

    p.select_one(HoverTool).tooltips = [
        (x_name, '@x_element'),
        (y_name, '@y_element'),
        (val_name, '@value')
    ]

    return p


def _populate_data_source(
    value_matrix, 
    x_labels, 
    y_labels, 
    norm_axis,
    color_scheme):

    (n, p) = value_matrix.shape
    normalizer = np.sum(value_matrix, axis=norm_axis)
    normed = value_matrix / normalizer \
        if norm_axis == 0 else \
        ((value_matrix.T / normalizer).T
    color_mat = (normed * len(color_scheme)).astype(int)
    get_index = lambda v: int(len(colors)*(v - (v % 0.1)))
    x_element = []
    y_element = []
    value = []
    color = []

    for j in xrange(p):
        for i in xrange(n):
            x_element.append(x_labels[j])
            y_element.append(y_labels[i])
            value.append(value_matrix[i,j])
            color.append(color_scheme[color_mat[i,j]])

    return ColumnDataSource(
        data=dict(
            y_element=y_element,
            x_element=x_element,
            color=color, 
            value=value)
    )
