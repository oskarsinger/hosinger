import numpy as np
import linal.utils.misc as lum

from math import pi
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.charts import HeatMap
from bokeh.palettes import YlGn9, OrRd9
from bokeh.models.layouts import Row
from utils import get_plot_path

def plot_matrix_heat(
    value_matrix, x_labels, 
    y_labels, 
    title,
    x_name,
    y_name,
    val_name,
    pos_color_scheme=None,
    neg_color_scheme=None,
    norm_axis=None,
    width=None,
    height=None,
    do_phase=False):

    if pos_color_scheme is None:
        pos_color_scheme = list(reversed(YlGn9))

    if neg_color_scheme is None:
        neg_color_scheme = list(reversed(OrRd9))

    (n, m) = value_matrix.shape

    if height is None:
        height = n * 30

    if width is None:
        width = m * 30

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

    for k, matrix in value_matrices.items():
        source = _populate_data_source(
            matrix, 
            x_labels, 
            y_labels,
            norm_axis,
            pos_color_scheme,
            neg_color_scheme)
        p = _initialize_figure(
            source,
            width,
            height,
            title + ' ' + k,
            x_labels,
            y_labels,
            x_name,
            y_name,
            val_name)

        ps.append(p)

    return Row(*ps)

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
        x_axis_location="above", toolbar_location="right", 
        tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '15pt'
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
    pos_color_scheme,
    neg_color_scheme):

    (n, p) = value_matrix.shape
    color_matrix = _get_color_matrix(
        value_matrix,
        norm_axis,
        len(pos_color_scheme))

    x_element = []
    y_element = []
    values = []
    color = []

    for j in xrange(p):
        for i in xrange(n):
            x_element.append(x_labels[j])
            y_element.append(y_labels[i])
            values.append(value_matrix[i,j])

            value_color = None

            if values[-1] >= 0:
                value_color = pos_color_scheme[color_matrix[i,j]]
            else:
                value_color = neg_color_scheme[color_matrix[i,j]]

            color.append(value_color)

    return ColumnDataSource(
        data=dict(
            y_element=y_element,
            x_element=x_element,
            color=color, 
            value=values))

def _get_color_matrix(
    matrix, norm_axis, num_colors):

    matrix = np.absolute(matrix)

    if norm_axis is None:
        normalizer = np.max(matrix)
        if normalizer == 0:
            normalizer = 1
    else:
        normalizer = np.max(
            matrix,
            axis=norm_axis)
        normalizer[normalizer == 0] = 1

    normed = matrix / normalizer \
        if norm_axis == 0 else \
        (matrix.T / normalizer).T
    pre_truncd = normed * (num_colors - 1)

    return pre_truncd.astype(int)
