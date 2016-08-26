import numpy as np

import filters

def dtwavexfm(
        X, nlevels, get_biort, get_qshift):

    (Yl, Yh, Y_scale) = (None, None, None)

    # TODO: make the proper calls to get_biort and get_qshift 
    (h0a, h0b, h1a, h1b, h0o, h1o) = (
        None, None, None, None, None, None)

    L = X.shape

    if L[0] % 2 > 0:
        raise ValueError(
            'Size of X must be a multiple of 2!')

    if nlevels == 0:
        return (Yl, Yh, Y_scale)

    Yh = [None] * nlevels
    Y_scale = [None] * nlevels
    j = 1j
    Hi = filters.get_column_filtered(X, h1o)
    Lo = filters.get_column_filtered(X, h0o)
    t = np.arange(0, Hi.shape[0]-1, 2)
    Yh[0] = Hi[t,:] + j * Hi[t+1,:]
    Y_scale[0] = np.copy(Lo)

    if nlevels >= 2:
        for level in range(1, nlevels-1):

            if Lo.shape[0] % 4 > 0:
                # TODO: finish up this nasty concatenation
                Lo = np.vstack(
                    np.copy(Lo[0,:]), 
                    something, 
                    something_else)

            Hi = filters.get_column_d_filtered(Lo, h1b, h1a)
            Lo = filters.get_column_d_filtered(Lo, h0b, h0a)
            t = np.arange(0, Hi.shape[0] - 1, 2)
            Yh[level] = Hi[t,:] + j * Hi[t+1,:]
            Y_scale[level] = np.copy(Lo)

    Yl = np.copy(Lo)

    return (Yl, Yh, Y_scale)

def dtwaveifm(
    Yl, Yh, get_biort, get_qshift, 
    gain_mask=None):

    a = Yh.shape[0]

    if gain_mask is None:
        gain_mask = np.ones((1,a))

    # TODO: make the proper calls to get_biort and get_qshift 
    (g0a, g0b, g1a, g1b, g0o, g1o) = (None, None, None, None, None, None)
    level = a - 1
    Lo = Yl

    while level >= 1:
        Hi = c2q1d(Yh[level] * gain_mask[level])
        Lo = filters.get_column_i_filtered(Lo, g0b, g0a) + \
            filters.get_column_i_filtered(Hi, g1b, g1a)

        (Lo_n, Lo_p) = Lo.shape
        (Yh_n, Yh_p) = Yh[level-1].shape

        if not Lo_n == 2 * Yh_n:
            Lo = Lo[1:Lo.shape[0]-2,:]

        if not (Lo_n == 2 * Yh_n and Lo_p == Yh_p):
            raise ValueError(
                'Yh sizes are not valid for DTWAVEIFM')

        level -= 1

    Hi = c2q1d(Yh[level] * gain_mask[level])

    return filters.get_column_filtered(Lo, g0o) + \
        filters.get_column_filtered(Hi, g1o)

def c2q1d(X):

    (n, p) = X.shape
    z = np.zeros((n*2, p))
    skip = np.arange(0, a*2-1, 2)
    z[skip,:] = np.real(x)
    z[skip+1,:] = np.imag(x)

    return z
