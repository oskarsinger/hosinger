import click

import wavelets.dtcwt as wdtcwt
import numpy as np

@click.command()
def run_it_all_day():

    X = np.random.randn(512,1)

    with open('data.txt', 'w') as f:
        X_str = '\n'.join([
            str(x) for x in X.T[0].tolist()])
        f.write(X_str)

    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')

    (Yl, Yh, Y_scale) = wdtcwt.oned.dtwavexfm(
        X, 5, near_sym_b, qshift_b)
    Z = wdtcwt.oned.dtwaveifm(
        Yl, Yh, near_sym_b, qshift_b)
    error = np.abs(Z-X).max()

    if error < 10**(-12):
        print 'Ur gud, dood'
    else:
        print error

    #TODO: implement the 2D test

if __name__=='__main__':
    run_it_all_day()
