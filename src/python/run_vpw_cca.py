import click

from runners.wavelets import ViewPairwiseCCARunner as VPWCCAR
from runners.wavelets import MVDTCWTRunner

@click.command()
@click.option('--data-path', default=None)
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--dataset', default='e4')
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--subject-mean', default=False)
@click.option('--nnz', default=1)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    dataset,
    save,
    load,
    show,
    subject_mean,
    nnz):

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
        subject_mean=subject_mean,
        nnz=nnz)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
