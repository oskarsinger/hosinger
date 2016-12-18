import click

from runners.wavelets import DTCWTPartialReconstructionRunner as DTCWTPRR
from runners.wavelets import MVDTCWTRunner

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir')
@click.option('--dataset', default='e4')
@click.option('--wavelet-dir')
@click.option('--missing', default=False)
@click.option('--complete', default=False)
@click.option('--save', default=False)
@click.option('--show', default=False)
@click.option('--avg', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    dataset,
    wavelet_dir,
    missing,
    complete,
    save,
    show,
    avg):

    dtcwt_runner = MVDTCWTRunner(
        data_path=data_path,
        dataset=dataset,
        save_load_dir=wavelet_dir,
        load=True)

    if not load:
        dtcwt_runner.run()

    runner = DTCWTPRR(
        dtcwt_runner,
        save_load_dir,
        missing=missing,
        complete=complete,
        save=save,
        show=show,
        avg=avg)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
