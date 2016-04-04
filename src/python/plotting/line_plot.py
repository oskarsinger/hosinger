import numpy as np

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Spectral11

from utils import get_plot_path

def plot_lines(
    data_map,
    x_label,
    y_label,
    title,
    colors=Spectral11
    width=900,
    height=400):

    p = figure(
        title=title,
        plot_width=width, plot_height=height,
        tools="resize,hover,save")
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
