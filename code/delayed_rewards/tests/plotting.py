from math import pi

from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, show, output_file
from bokeh.palettes import Spectral10

def get_arms_vs_time(bandit, filename=None):

    status = bandit.get_status()
    history = status['history']
    actions = status['actions']
    times = list(range(len(history)))

    # Set up the data for plotting. We will need to have values for every
    # pair of year/month names. Map the rate to a color.
    action_counts = {action : 0 for action in actions}
    action = []
    time = []
    color = []
    rate = []

    for (i, (a_i, r, p)) in enumerate(history):
        action_counts[a_i] += 1

        rates = {key : float(count)/float(i)
                 for key, count in action_counts.items()}

        for a in actions:
            action.append(a)
            time.append(i) 
            rate_a_i = rates[a]
            rate.append(rates[a])
            index = int(10*(rate_a_i - (rate_a_i % 10)))
            color.append(Spectral10[index])

    source = ColumnDataSource(
        data=dict(action=action, time=time, color=color, rate=rate)
    )

    TOOLS = "resize,hover,save,pan,box_zoom,wheel_zoom"

    p = figure(title="Percentage of total pullage per arm over time",
               x_range=times, y_range=list(reversed(actions)),
               x_axis_location="above", plot_width=900, plot_height=400,
               toolbar_location="left", tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi/3

    p.rect("time", "action", 1, 1, source=source,
           color="color", line_color=None)

    p.select_one(HoverTool).tooltips = [
        ('time and action', '@time @action'),
        ('rate', '@rate'),
    ]

    output_file('pullage.html', title="Pullage per arm over time")

    show(p)      # show the plot 
