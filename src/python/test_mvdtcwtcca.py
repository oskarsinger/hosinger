import click

import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt

from runners.multiview import MVCCADTCWTRunner
from data.servers.batch import BatchServer as BS
from bokeh.plotting import show
from bokeh.models.layouts import Column

@click.command()
@click.option('--data-path', 
    default='/home/oskar/Data/VirusGenomeData/FullE4/20160503_BIOCHRON_E4.hdf5')
@click.option('--save-heat', default=False)
@click.option('--load-heat', default=False)
@click.option('--heat-dir', default=None)
@click.option('--show-plots', default=False)
@click.option('--center', default=False)
@click.option('--period', default=24*3600)
def run_it_all_day(
    data_path, 
    save_heat, 
    load_heat,
    heat_dir, 
    show_plots,
    center, 
    period):
    print heat_dir

    # TODO: do it with different bases and shifts
    # TODO: also figure out what shifts are
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')
    loaders = dles.get_changing_e4_loaders(
        data_path,
        'HRV15-005',
        None,
        False)
    servers = [BS(dl, center=center) for dl in loaders]
    runner = MVCCADTCWTRunner(
        near_sym_b,
        qshift_b,
        servers,
        period,
        heat_dir=heat_dir,
        save_heat=save_heat,
        load_heat=load_heat,
        show_plots=show_plots)

    runner.run()

if __name__=='__main__':
    run_it_all_day()
