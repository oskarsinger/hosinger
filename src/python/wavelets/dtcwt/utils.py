import numpy as np

import os

def reflect(X, minx, maxx):

    print 'X', X
    Y = np.copy(X)
    t = np.nonzero(Y > maxx)
    Y[t] = 2 * maxx - Y[t]

    t = np.nonzero(Y < minx)

    while np.count_nonzero(t) > 0:
        Y[t] = 2 * minx - Y[t]
        t = np.nonzero(Y > maxx)

        if not np.count_nonzero(t) > 0:
            Y[t] = 2 * maxx - Y[t]

        t = np.nonzero(Y < minx)

    return Y - 1

def get_wavelet_basis(wavelet_name):

    path_items = [os.pardir] * 2 + [
        'constants', 
        'wavelets',
        wavelet_name + '.csv']
    path = os.path.join(*path_items)

    with open(path) as f:
        titles = [t.strip() 
                  for t in f.readline().split(',')]
        val_lists = {t : [] for t in titles}

        for l in f:
            vals = l.split(',')

            for (t, v) in zip(titles, vals):
                if len(v) > 0:
                    val_lists[t].append(float(v))

    return {t : np.array(vec)[:,np.newaxis]
            for (t, vec) in val_lists.items()}
