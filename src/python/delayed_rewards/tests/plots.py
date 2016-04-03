import os

import numpy as np

from plotting import plot_matrix_heat
from plotting.utils import get_plot_path

from bokeh.plotting import figure, show, output_file, vplot

def plot_arms_vs_time(bandit):

    status = bandit.get_status()
    history = status['history']
    actions = [str(a) for a in status['actions']]
    times = [str(i) for i in range(len(history))]
    action_counts = {action : 0 for action in actions}
    counts = np.zeros((len(actions), len(times)))

    for (i, (a_i, r, p)) in enumerate(history):
        action_counts[str(a_i)] += 1

        for a in actions:
            counts[a, i] = action_counts[a]

    p = plot_matrix_heat(
        counts,
        times,
        actions,
        'Percent pullage per arm over time',
        'time',
        'action',
        'pullage') 
    filepath = os.path.join(
        get_plot_path(), 
        'pullage_per_arm_over_time.html')

    output_file(filepath, title='percent pullage per arm over time')
    show(p)
