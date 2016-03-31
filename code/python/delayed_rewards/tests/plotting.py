from plotting import plot_matrix_heat

def plot_arms_vs_time(bandit, filename=None):

    if filename is None:
        filename = 'pullage.html'

    status = bandit.get_status()
    history = status['history']
    actions = [str(a) for a in status['actions']]
    times = [str(i) for i in range(len(history))]

    # Set up the data for plotting. We will need to have values for every
    # pair of year/month names. Map the rate to a color.
    action_counts = {action : 0 for action in actions}
    action = []
    time = []
    rate = []

    for (i, (a_i, r, p)) in enumerate(history):
        action_counts[str(a_i)] += 1

        rates = {key : float(count)/float(i+1)
                 for key, count in action_counts.items()}

        for a in actions:
            action.append(a)
            time.append(i) 
            rate_a_i = rates[a]
            rate.append(rate_a_i)

    plot_matrix_heat(
        action,
        time,
        rates,
        'Percent pullage per arm over time',
        'action',
        'time',
        '%pullage') 
