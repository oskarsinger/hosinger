import click

from runners.multiview import MVCCADTCWTRunner

@click.command()
@click.option('')
def run_it_all_day():

    # TODO: do it with different bases and shifts
    # TODO: also figure out what shifts are
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')

    runner = 

if __name__=='__main__':
    run_it_all_day()
