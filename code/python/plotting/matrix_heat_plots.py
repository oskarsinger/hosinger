from math import pi

from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, show, output_file
from bokeh.palettes import Spectral10

def plot_matrix_heat(
    y_elements, 
    x_elements, 
    values,
    title,
    y_name,
    x_name,
    val_name,
    filename=None,
    width=900,
    height=400):

    if not all([value >= 0 and value <= 1 for value in values]):
        raise ValueError(
            'Elements of values must all be in closed interval [0,1].')

    if filename is None:
        filename = '_'.join([
            val_name,
            'of',
            y_name,
            'per',
            x_name,
            'matrix',
            'heat',
            'plot']) + '.html'

    get_index = lambda value: int(10*(value - (value % 0.1)))
    color = [Spectral10[get_index(value)]
             for value in values]
    source = ColumnDataSource(
        data=dict(
            y_elements=y_elements, 
            x_elements=x_elements, 
            color=color, 
            values=values)
    )

    TOOLS = "resize,hover,save,pan,box_zoom,wheel_zoom"

    p = figure(title=title,
        x_range=times, y_range=list(reversed(y_elements)),
        x_axis_location="above", plot_width=900, plot_height=400,
        toolbar_location="left", tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi/3

    p.rect(x_name, y_name, 1, 1, source=source,
           color='color', line_color=None)

    p.select_one(HoverTool).tooltips = [
        (x_name + ' and ' + y_name, '@'+ x_name + ' @' + y_name),
        (val_name, '@' + val_name),
    ]

    output_file(filename, title=title)

    show(p)      # show the plot 
