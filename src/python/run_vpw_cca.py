import click

from runners.multiview import ViewPairwiseCCARunner as VPWCCAR
from runners.multiview import MVDTCWTRunner

@click.command()
@click.option('--data-path', default=None)
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--dataset', default='e4')
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--show-mean', default=False)
@click.option('--subject-mean', default=False)
@click.option('--show-transpose', default=False)
@click.option('--show-cc', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    dataset,
    save,
    load,
    show,
    show_mean,
    subject_mean,
    show_transpose,
    show_cc):

    dtcwt_runner = MVDTCWTRunner(
        data_path=data_path,
        dataset=dataset,
        save_load_dir=wavelet_dir,
        load=True)

    if not load:
        dtcwt_runner.run()

    runner = VPWCCAR(
        dtcwt_runner,
        save_load_dir,
        save=save,
        load=load,
        show=show,
        show_mean=show_mean,
        subject_mean=subject_mean,
        show_transpose=show_transpose,
        show_cc=show_cc)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
