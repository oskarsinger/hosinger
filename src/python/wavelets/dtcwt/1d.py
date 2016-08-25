import numpy as np

import filters

def dtwaveifm(
    Yl, Yh, get_biort, get_qshift, 
    gain_mask=None):

    a = Yh.shape[0]

    if gain_mask is None:
        gain_mask = np.ones((1,a))

    # TODO: make the proper calls to get_biort and get_qshift 
    (g0a, g0b, g1a, g1b, g0o, g1o) = (None, None, None, None, None, None)

    levels = a

    Lo = Yl

    while level >= 2:
        Hi = c2q1d(Yh[level] * gain_mask[level])
        Lo = filters.get_column_i_filtered(Lo, g0b, g0a) + \
            filters.get_column_i_filtered(Hi, g1b, g1a)

        (Lo_n, Lo_p) = Lo.shape
        (Yh_n, Yh_p) = Yh[level-1].shape

        if not Lo_n == 2 * Yh_n:
            Lo = Lo[2:Lo.shape[0]-1,:]

        if not (Lo_n == 2 * Yh_n and Lo_p == Yh_p):
            raise ValueError(
                'Yh sizes are not valid for DTWAVEIFM')

        level = level - 1

    Hi = c2q1d(Yh[level] * gain_mask[level])

    return filters.get_column_filtered(Lo, g0o) + \
        filters.get_column_filtered(Hi, g1o)

def c2q1d(X):

    (n, p) = X.shape
    z = np.zeros((n*2, p))
    skip = np.arange(1, a*2, 2)
    z[skip,:] = np.real(x)
    z[skip+1,:] = np.imag(x)

    return z
