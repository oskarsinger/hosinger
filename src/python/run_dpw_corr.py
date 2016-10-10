import click

from runners.multiview import DayPairwiseCorrelationRunner as DPCR

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    save,
    load,
    show):

    dtcwt_runner = MVDTCWTRunner(
        data_path,
        save_load_dir=wavelet_dir,
        load=True)

    dtcwt_runner.run()

    runner = DPCR(
        dtcwt_runner,
        save_load_dir,
        save=save,
        load=load,
        show=show)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
