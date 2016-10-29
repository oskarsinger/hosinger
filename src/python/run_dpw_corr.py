import click

from runners.multiview import DayPairwiseCorrelationRunner as DPCR
from runners.multiview import MVDTCWTSPRunner

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--test-data', default=False)
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--show-max', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    test_data,
    save,
    load,
    show,
    show_max):

    dtcwt_runner = MVDTCWTSPRunner(
        data_path,
        test_data=test_data,
        save_load_dir=wavelet_dir,
        load=True)

    if not load:
        dtcwt_runner.run()

    runner = DPCR(
        dtcwt_runner,
        save_load_dir,
        save=save,
        load=load,
        show=show,
        show_max=show_max)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
