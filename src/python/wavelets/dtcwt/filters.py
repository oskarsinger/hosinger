import numpy as np

from scipy.signal import convolve2d as conv2
import utils as dtcwtu

# TODO: double check indexing to translate from Matlab to numpy

def get_column_filtered(X, h):

    (n, p) = X.shape
    m = h.shape[0]
    m2 = int(m/2)
    Y = None
    
    if np.count_nonzero(X) > 0:
        xe = dtcwtu.reflect(np.arange(1-m2, n+m2+1), 0.5, n+0.5)

        Y = conv2(X[xe,:], h, mode='valid')
    else:
        Y = np.zeros((n+1-(m % 2), p))

    return Y

def get_column_i_filtered(X, ha, hb):

    (n, p) = X.shape

    if n % 2 > 0:
        raise ValueError(
            'No. of rows in X must be multiple of 2!')

    m = ha.shape[0]

    if not m == hb.shape[0]:
        raise ValueError(
            'Lengths of ha and hb must be the same!')

    if m % 2 > 0:
        raise ValueError(
            'Lengths of ha and hb must be even!')

    m2 = int(m/2)
    Y = np.zeros((n*2, p))

    xe = dtcwtu.reflect(np.arange(1-m2, n+m2+1), 0.5, n+0.5)
    hao = ha[0:m:2]
    hae = ha[1:m:2]
    hbo = hb[0:m:2]
    hbe = hb[1:m:2]
    s = np.arange(0, n*2, 4)

    if m2 % 2 == 0:

        t = np.arange(3, n+m, 2)
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

        t = np.arange(2, n+m-1, 2)
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

def get_column_d_filtered(X, ha, hb):

    (n, p) = X.shape

    if n % 4 > 0:
        raise ValueError(
            'No. of rows in X must be a multiple of 4!')

    m = ha.shape[0]

    if not m == hb.shape[0]:
        raise ValueError(
            'Lengths of ha and hb must be the same!')

    if m % 2 > 0:
        raise ValueError(
            'Lengths of ha and hb must be even!')

    m2 = int(m/2)

    xe = dtcwtu.reflect(np.arange(1-m,n+m), 0.5, n+0.5)
    print 'xe\n', xe

    hao = ha[0:m:2]
    hae = ha[1:m:2]
    hbo = hb[0:m:2]
    hbe = hb[1:m:2]
    t = np.arange(5, n+2*m-2, 4)
    print 't\n', t

    n2 = n/2
    Y = np.zeros((n2, p))
    begin2 = 0
    end2 = n2
    begin1 = 1
    end1 = n2+1

    if np.sum(ha * hb) > 0:
        begin2 = 1
        end2 = n2+1
        begin1 = 0
        end1 = n2

    print 'xe[t-1]\n', xe[t-1]
    Y[begin1:end1:2,:] = conv2(X[xe[t-1],:], hao, 'valid') + \
        conv2(X[xe[t-3],:], hae, 'valid')
    Y[begin2:end2:2,:] = conv2(X[xe[t],:], hbo, 'valid') + \
        conv2(X[xe[t-2],:], hbe, 'valid')

    return Y
