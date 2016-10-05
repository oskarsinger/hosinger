import click
import h5py

import data.loaders.e4.shortcuts as dles
import wavelets.dtcwt as wdtcwt

from runners.multiview import MVCCADTCWTRunner
from data.servers.batch import BatchServer as BS
from bokeh.plotting import show
from bokeh.models.layouts import Column

@click.command()
@click.option('--data-path', 
    default='/home/oskar/Data/VirusGenomeData/FullE4/20160503_BIOCHRON_E4.hdf5')
@click.option('--delay', default=None)
@click.option('--correlation-dir', default=None)
@click.option('--load-correlation', default=False)
@click.option('--save-correlation', default=False)
@click.option('--show-correlation', default=False)
@click.option('--cca-dir', default=None)
@click.option('--load-cca', default=False)
@click.option('--save-cca', default=False)
@click.option('--show-cca', default=False)
@click.option('--center', default=False)
@click.option('--period', default=12*3600)
@click.option('--kmeans', default=None)
@click.option('--plot-dir', default='../../plots/')
def run_it_all_day(
    data_path, 
    delay,
    correlation_dir, 
    load_correlation,
    save_correlation, 
    show_correlation,
    cca_dir, 
    load_cca,
    save_cca, 
    show_cca,
    center, 
    period,
    kmeans,
    plot_dir):

    # TODO: do it with different bases and shifts
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')
    runner = MVCCADTCWTRunner(
        near_sym_b,
        qshift_b,
        period,
        delay=delay,
        correlation_dir=correlation_dir,
        save_correlation=save_correlation,
        load_correlation=load_correlation,
        show_correlation=show_correlation,
        cca_dir=cca_dir,
        save_cca=save_cca,
        load_cca=load_cca,
        show_cca=show_cca,
        kmeans=kmeans,
        plot_dir=plot_dir)

    runner.run()

if __name__=='__main__':
    run_it_all_day()
