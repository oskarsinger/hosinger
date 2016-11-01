import click

from runners.multiview import ViewPairwiseCorrelationRunner as VPWCR
from runners.multiview import MVDTCWTRunner

@click.command()
@click.option('--data-path', default=None)
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--dataset', default='e4')
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--show-max', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    dataset,
    save,
    load,
    show,
    show_max):

    dtcwt_runner = MVDTCWTRunner(
        data_path=data_path,
        dataset=dataset,
        save_load_dir=wavelet_dir,
        load=True)

    if not load:
        dtcwt_runner.run()

    runner = VPWCR(
        dtcwt_runner,
        save_load_dir,
        save=save,
        load=load,
        show=show,
        show_max=show_max)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
