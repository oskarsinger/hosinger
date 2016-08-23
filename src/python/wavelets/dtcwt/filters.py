import numpy as np

from scipy.signal import convolve2 as conv2

def get_column_filter(X, h):

    (n, p) = X.shape
    m = h.shape[0]
    m2 = int(m/2)
    Y = None
    
    if np.count_nonzero(X) > 0:
        # TODO: figure out what the 0.5 and r+0.5 in the matlab reflect call are
        # TODO: translate those to probably padding length and maybe other args
        xe = np.pad(np.arange(1-m2, n+m2), something, 'reflect')

        Y = conv2(X[xe,:], h, mode='valid')
    else:
        Y = np.zeros(n+1-(m % 2), p)

    return np.copy(Y)

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

    # TODO: figure out what the 0.5 and r+0.5 in the matlab reflect call are
    # TODO: translate those to probably padding length and maybe other args
    xe = np.pad(np.arange(1-m2, n+m2), something, 'reflect')
    hao = ha[1:m:2]
    hae = ha[2:m:2]
    hbo = hb[1:m:2]
    hbe = hb[2:m:2]
    s = np.arange(1, n*2, 4)

    if m2 % 2 == 0:

        t = np.arange(4, n+m, 2)
        ta = t - 1
        tb = t

        if np.sum(ha * hb) > 0:
            ta = t
            tb = t - 1

        Y[s,:] = conv2(X[xe[tb-2],:], hae, 'valid')
        Y[s+1,:] = conv2(X[xe[ta-2],:], hbe, 'valid')
        Y[s+2,:] = conv2(X[xe[tb],:], hao, 'valid')
        Y[s+3,:] = conv2(X[xe[ta],:], hbo, 'valid')
    else:

        t = np.arange(3, n+m-1, 2)
        ta = t - 1
        tb = t

        if np.sum(ha * hb) > 0:
            ta = t
            tb = t - 1

        Y[s,:] = conv2(X[xe[tb],:], hao, 'valid')
        Y[s+1,:] = conv2(X[xe[ta],:], hbo, 'valid')
        Y[s+2,:] = conv2(X[xe[tb],:], hae, 'valid')
        Y[s+3,:] = conv2(X[xe[ta],:], hbe, 'valid')

    return np.copy(Y)
