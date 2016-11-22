import click

from runners.multiview import E4DTCWTPartialReconstructionRunner as E4DTCWTPRR
from runners.multiview import MVDTCWTRunner

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--missing', default=False)
@click.option('--complete', default=False)
@click.option('--std', default=False)
@click.option('--save', default=False)
@click.option('--load', default=False)
@click.option('--show', default=False)
@click.option('--avg', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    missing,
    complete,
    std,
    save,
    load,
    show,
    avg):

    dtcwt_runner = MVDTCWTSPRunner(
        data_path,
        save_load_dir=wavelet_dir,
        load=True)

    if not load:
        dtcwt_runner.run()

    runner = E4DTCWTPRR(
        dtcwt_runner,
        save_load_dir,
        missing=missing,
        complete=complete,
        std=std,
        save=save,
        load=load,
        show=show,
        avg=avg)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
