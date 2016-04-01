import numpy as np

from plotting import plot_matrix_heat

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

    plot_matrix_heat(
        counts,
        actions,
        times,
        'Percent pullage per arm over time',
        'action',
        'time',
        'pullage') 
