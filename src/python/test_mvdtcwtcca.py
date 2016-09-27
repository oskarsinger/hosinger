import click

import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt

from runners.multiview import MVCCADTCWTRunner
from data.servers.batch import BatchServer as BS
from bokeh.plotting import show
from bokeh.models.layouts import Column

@click.command()
@click.argument('center', default=True)
@click.argument('period', default=24*3600)
def run_it_all_day(center, period):

    # TODO: do it with different bases and shifts
    # TODO: also figure out what shifts are
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')
    loaders = dles.get_changing_e4_loaders(
        '/home/oskar/Data/VirusGenomeData/FullE4/20160503_BIOCHRON_E4.hdf5',
        'HRV15-005',
        None,
        False)
    servers = [BS(dl, center=center) for dl in loaders]
    runner = MVCCADTCWTRunner(
        near_sym_b,
        qshift_b,
        2,
        servers,
        period)
    heat_plots = runner.run()

    plots = heat_plots[0].values()[0].values()
    show(Column(*plots))

if __name__=='__main__':
    run_it_all_day()
