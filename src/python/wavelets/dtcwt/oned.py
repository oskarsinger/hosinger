import numpy as np

import filters

def dtwavexfm(
    X, nlevels, biorthogonal, q_shift):

    (Yl, Yh, Y_scale) = [None] * 3

    (h0a, h0b, h1a, h1b, h0o, h1o) = (
        q_shift['h0a'],
        q_shift['h0b'],
        q_shift['h1a'],
        q_shift['h1b'],
        biorthogonal['h0o'],
        biorthogonal['h1o'])

    L = X.shape

    if L[0] % 2 > 0:
        print L[0]
        raise ValueError(
            'Size of X must be a multiple of 2!')

    if nlevels == 0:
        return (Yl, Yh, Y_scale)

    Yh = [None] * nlevels
    Y_scale = [None] * nlevels
    j = 1j
    print 'Initializing highpass filter'
    Hi = filters.get_column_filtered(X, h1o)
    print 'Initializing lowpass filter'
    Lo = filters.get_column_filtered(X, h0o)
    t = np.arange(0, Hi.shape[0]-1, 2)
    Yh[0] = Hi[t,:] + j * Hi[t+1,:]
    Y_scale[0] = np.copy(Lo)

    if nlevels >= 2:
        print 'First loop running for', nlevels-2, 'rounds'
        for level in range(1, nlevels-1):

            if Lo.shape[0] % 4 > 0:
                Lo = np.vstack(
                    np.copy(Lo[0,:]), 
                    np.copy(Lo), 
                    np.copy(Lo[-1,:])).T

            print 'Getting highpass filter for round', level
            Hi = filters.get_column_d_filtered(Lo, h1b, h1a)
            print 'Getting lowpass filter for round', level
            Lo = filters.get_column_d_filtered(Lo, h0b, h0a)
            t = np.arange(0, Hi.shape[0] - 1, 2)
            Yh[level] = Hi[t,:] + j * Hi[t+1,:]
            Y_scale[level] = np.copy(Lo)

    Yl = np.copy(Lo)

    return (Yl, Yh, Y_scale)

def dtwaveifm(
    Yl, Yh, biorthogonal, q_shift, 
    gain_mask=None):

    a = len(Yh)

    if gain_mask is None:
        gain_mask = np.ones((1,a))

    (g0a, g0b, g1a, g1b, g0o, g1o) = (
        q_shift['g0a'],
        q_shift['g0b'],
        q_shift['g1a'],
        q_shift['g1b'],
        biorthogonal['g0o'],
        biorthogonal['g1o'])
        
    level = a - 1
    Lo = Yl

    while level >= 1:
        Hi = c2q1d(Yh[level] * gain_mask[level])
        Lo = filters.get_column_i_filtered(Lo, g0b, g0a) + \
            filters.get_column_i_filtered(Hi, g1b, g1a)

        (Lo_n, Lo_p) = Lo.shape
        (Yh_n, Yh_p) = Yh[level-1].shape

        if not Lo_n == 2 * Yh_n:
            Lo = Lo[1:Lo.shape[0]-1,:]

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
