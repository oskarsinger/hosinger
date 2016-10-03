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
@click.option('--subject', default=None)
@click.option('--delay', default=None)
@click.option('--save-correlation', default=False)
@click.option('--load-correlation', default=False)
@click.option('--correlation-dir', default=None)
@click.option('--show-correlation', default=False)
@click.option('--save-cca', default=False)
@click.option('--load-cca', default=False)
@click.option('--cca-dir', default=None)
@click.option('--show-cca', default=False)
@click.option('--center', default=False)
@click.option('--period', default=12*3600)
@click.option('--kmeans', default=None)
@click.option('--plot-path', default='../../plots/')
def run_it_all_day(
    data_path, 
    subject,
    delay,
    save_correlation, 
    load_correlation,
    correlation_dir, 
    show_correlation,
    save_cca, 
    load_cca,
    cca_dir, 
    show_cca,
    center, 
    period,
    plot_path):

    # TODO: do it with different bases and shifts
    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')

    if subject is None:
        subjects = h5py.File(data_path).keys()

        for subject in subjects:
            try:
                loaders = dles.get_hr_and_acc(
                    data_path,
                    subject,
                    None,
                    False)
                servers = [BS(dl, center=center) for dl in loaders]
                runner = MVCCADTCWTRunner(
                    near_sym_b,
                    qshift_b,
                    servers,
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
                    plot_path=plot_path)

                runner.run()
            except Exception, e:
                print e
    else:
        loaders = dles.get_hr_and_acc(
            data_path,
            subject,
            None,
            False)
        servers = [BS(dl, center=center) for dl in loaders]
        runner = MVCCADTCWTRunner(
            near_sym_b,
            qshift_b,
            servers,
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
            plot_path=plot_path)

        runner.run()

if __name__=='__main__':
    run_it_all_day()
