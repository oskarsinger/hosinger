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
@click.option('--save-load-dir', default='.')
@click.option('--compute-correlation', default=False)
@click.option('--load-correlation', default=False)
@click.option('--save-correlation', default=False)
@click.option('--show-correlation', default=False)
@click.option('--compute-cca', default=False)
@click.option('--load-cca', default=False)
@click.option('--save-cca', default=False)
@click.option('--show-cca', default=False)
@click.option('--center', default=False)
@click.option('--period', default=24*3600)
@click.option('--sub-period', default=3600)
@click.option('--correlation-kmeans', default=None)
@click.option('--cca-kmeans', default=None)
def run_it_all_day(
    data_path, 
    delay,
    save_load_dir,
    compute_correlation,
    load_correlation,
    save_correlation, 
    show_correlation,
    compute_cca,
    load_cca,
    save_cca, 
    show_cca,
    center, 
    period,
    sub_period,
    correlation_kmeans,
    cca_kmeans):

    if correlation_kmeans is not None:
        correlation_kmeans = int(correlation_kmeans)

    if cca_kmeans is not None:
        cca_kmeans = int(cca_kmeans)

    if delay is not None:
        delay = int(delay)

    # TODO: do it with different bases and shifts
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')
    runner = MVCCADTCWTRunner(
        data_path,
        near_sym_b,
        qshift_b,
        period,
        sub_period,
        delay=delay,
        save_load_dir=save_load_dir,
        compute_correlation=compute_correlation,
        save_correlation=save_correlation,
        load_correlation=load_correlation,
        show_correlation=show_correlation,
        compute_cca=compute_cca,
        save_cca=save_cca,
        load_cca=load_cca,
        show_cca=show_cca,
        correlation_kmeans=correlation_kmeans,
        cca_kmeans=cca_kmeans)

    runner.run()

if __name__=='__main__':
    run_it_all_day()
