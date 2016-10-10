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
@click.option('--compute-wavelets', default=False)
@click.option('--save-wavelets', default=False)
@click.option('--compute-sp-wavelets', default=False)
@click.option('--save-sp-wavelets', default=False)
@click.option('--compute-sp-corr', default=False)
@click.option('--load-sp-corr', default=False)
@click.option('--save-sp-corr', default=False)
@click.option('--show-sp-corr', default=False)
@click.option('--compute-vpw-corr', default=False)
@click.option('--load-vpw-corr', default=False)
@click.option('--save-vpw-corr', default=False)
@click.option('--show-vpw-corr', default=False)
@click.option('--compute-dpw-corr', default=False)
@click.option('--load-dpw-corr', default=False)
@click.option('--save-dpw-corr', default=False)
@click.option('--show-dpw-corr', default=False)
@click.option('--compute-vpw-cca', default=False)
@click.option('--load-vpw-cca', default=False)
@click.option('--save-vpw-cca', default=False)
@click.option('--show-vpw-cca', default=False)
@click.option('--center', default=False)
@click.option('--period', default=24*3600)
@click.option('--subperiod', default=None)
@click.option('--do-phase', default=False)
@click.option('--sp-corr-kmeans', default=None)
@click.option('--vpw-corr-kmeans', default=None)
@click.option('--vpw-cca-kmeans', default=None)
@click.option('--show-kmeans', default=False)
@click.option('--show-vpw-corr-subblocks', default=False)
def run_it_all_day(
    data_path, 
    delay,
    save_load_dir,
    compute_wavelets,
    save_wavelets,
    load_wavelets,
    compuet_sp_wavelets,
    save_sp_wavelets,
    load_sp_wavelets,
    compute_sp_corr,
    load_sp_corr,
    save_sp_corr, 
    show_sp_corr,
    compute_vpw_corr,
    load_vpw_corr,
    save_vpw_corr, 
    show_vpw_corr,
    compute_dpw_corr,
    load_dpw_corr,
    save_dpw_corr, 
    show_dpw_corr,
    compute_vpw_cca,
    load_vpw_cca,
    save_vpw_cca, 
    show_vpw_cca,
    center, 
    period,
    subperiod,
    do_phase,
    sp_corr_kmeans,
    vpw_corr_kmeans,
    vpw_cca_kmeans,
    show_kmeans,
    show_vpw_corr_subblocks)

    if correlation_kmeans is not None:
        correlation_kmeans = int(correlation_kmeans)

    if cca_kmeans is not None:
        cca_kmeans = int(cca_kmeans)

    if delay is not None:
        delay = int(delay)

    if subperiod is not None:
        subperiod = int(subperiod)

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
        subperiod=subperiod,
        do_phase=do_phase,
        delay=delay,
        save_load_dir=save_load_dir,
        compute_wavelets=compute_wavelets,
        save_wavelets=save_wavelets,
        compute_correlation=compute_correlation,
        save_correlation=save_correlation,
        load_correlation=load_correlation,
        show_correlation=show_correlation,
        compute_cca=compute_cca,
        save_cca=save_cca,
        load_cca=load_cca,
        show_cca=show_cca,
        correlation_kmeans=correlation_kmeans,
        cca_kmeans=cca_kmeans,
        show_kmeans=show_kmeans,
        show_corr_subblocks=show_corr_subblocks,
        show_sp_correlation=show_sp_correlation)

    runner.run()

if __name__=='__main__':
    run_it_all_day()
