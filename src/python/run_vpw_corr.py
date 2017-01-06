import click

from runners.wavelets import ViewPairwiseCorrelationRunner as VPWCR
from runners.wavelets import MVDTCWTRunner

@click.command()
@click.option('--data-path', default=None)
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--dataset', default='e4')
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--avg-over-subjects', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    dataset,
    save,
    load,
    show,
    avg_over_subjects):

    dtcwt_runner = MVDTCWTRunner(
        data_path=data_path,
        dataset=dataset,
        save_load_dir=wavelet_dir,
        load=True)

    dtcwt_runner.run()

    runner = VPWCR(
        dtcwt_runner,
        save_load_dir,
        save=save,
        load=load,
        show=show,
	avg_over_subjects=avg_over_subjects)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
