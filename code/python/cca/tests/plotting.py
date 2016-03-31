from math import log

from data.smart_watch import get_data_summaries

from bokeh.plotting import figure, show, output_file

def plot_action_counts(data_dir):

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
    scale = 10**(log(x_max), base=10)
    x_max = (x_max - x_max % scale) + scale
    p = figure(title="Action Counts", tools="resize,save", y_range=factors, x_range=[0,x_max])

    p.segment(0, factors, x, factors, line_width=2, line_color="green", )
    p.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

    output_file("action_counts.html", title="action counts")

    show(p)

def plot_canonical_bases(AppGradCCA):

    print "Some stuff"
