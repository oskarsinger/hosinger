import numpy as np

def get_column_filter(X, h):

    (n, p) = X.shape
    m = h.shape[0]
    m2 = int(m/2)
    Y = None
    
    if np.count_nonzero(X) > 0:
        # TODO: figure out what the 0.5 and r+0.5 in the matlab reflect call are
        # TODO: translate those to probably padding length and maybe other args
        xe = np.pad(np.arange(1-m2, n+m2), something, 'reflect')

        Y = np.convolve2(X[xe,:], h, mode='valid')
    else:
        Y = np.zeros(n+1-(m % 2), p)

    return Y

def get_column_i_filter(X, ha, hb):

    (n, p) = X.shape

    if r % 2 > 0:
        raise ValueError(
            'No. of rows in X must be multiple of 2!')

    m = ha.shape[0]

    if not m == hp.shape[0]:
        raise ValueError(
            'Lengths of ha and hb must be the same!')

    if m % 2 > 0:
        raise ValueError(
            'Lengths of ha and hb must be even!')

    m2 = int(m/2)
    Y = np.zeros(n*2, p)
