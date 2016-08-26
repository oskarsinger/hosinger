import numpy as np

import filters

def dtwaveifm2(
    Yl, Yh, biort, qshift, 
    gain_mask=None):

    a = Yh.shape[0]

    if gain_mask is None:
        gain_mask = np.ones(6, a)

    # TODO: make the proper calls to get_biort and get_qshift 
    (g0a, g0b, g1a, g1b, g0o, g1o) = (
        None, None, None, None, None, None)
        
    current_level = a - 1;

    Z = np.copy(Yl)
    get_c2q = lambda idxs,lvl: c2q(Yh, gain_mask, idxs, lvl)

    while current_level >= 1:


        lh = get_c2q(np.array([0,5]), current_level)
        hl = get_c2q(np.array([2,3]), current_level)
        hh = get_c2q(np.array([1,4]), current_level)

        # Do even Qshift filters on columns.
        y1 = filters.get_column_i_filtered(Z, g0b, g0a) + \
            filters.get_column_i_filtered(lh, g1b, g1a)
        y2 = filters.get_column_i_filtered(hl, g0b, g0a) + \
            filters.get_column_i_filtered(hh, g1b, g1a)

        # Do even Qshift filters on rows.
        ZT = filters.get_column_i_filtered(y1.T, g0b, g0a) + \
            filters.get_column_i_filtered(y2.T, g1b, g1a)
        Z = ZT.T

        # Check size of Z and crop as required
        (n, p) = Z.shape

        # TODO: fill this in

        current_level -= 1

    lh = get_c2q(np.array([0,5]), 0)
    hl = get_c2q(np.array([2,3]), 0)

def c2q(Yh, gain_mask, indexes, level):

    w = Yh[level][:,:,indexes]
    gain = gain_mask[indexes,level]

    sw = w.shape
    x = np.zeros((2*sw[0], 2*sw[1]))
    nonzeros = [np.count_nonzero(a)
                for a in [w, gain]]

    if sum(nonzeros) > 0:
        sc = 0.5**(0.5) * gain
        P = w[:,:,0] * sc[0] + w[:,:,1] * sc[1]
        Q = w[:,:,0] * sc[0] - w[:,:,1] * sc[1]

        t1 = np.arange(0, x.shape[0], 2)
        t2 = np.arange(0, x.shape[1], 2)

        x[t1, t2] = np.real(P)
        x[t1, t2+1] = np.imag(P)
        x[t1+1, t2] = np.image(Q)
        x[t1+1, t2+1] = -np.real(Q)

    return x
