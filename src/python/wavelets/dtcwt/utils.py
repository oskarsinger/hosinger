import numpy as np
import os

def reflect(X, minx, maxx):

    Y = np.copy(X)
    t = np.nonzero(Y > maxx)
    Y[t] = 2 * maxx - Y[t]

    t = np.nonzero(Y < minx)

    while not np.all(t==0):
        Y[t] = 2 * minx - Y[t]
        t = np.nonzero(Y > maxx)

        if not np.all(t==0):
            Y[t] = 2 * maxx - Y[t]

        t = np.nonzero(Y < minx)

    return Y

def get_wavelet_basis(wavelet_name):

    base_dir = '/'.join(
        ['..'] * 4 +
        ['constants', 'wavelets'])
    path = os.path.join([
        base_dir, 
        wavelet_name, 
        '.csv'])

    with open(path) as f:
        titles = f.readline().split(',')
        val_lists = {t : [] for t in titles}

        for l in f:
            vals = l.split(',')

            for (t, v) in zip(titles, vals):
                if len(v) > 0:
                    val_lists[t].append(float(v))

    return {t : np.array(vec)
            for (t, vec) in val_lists.items()}
