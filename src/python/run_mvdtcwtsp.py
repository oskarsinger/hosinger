import click

from runners.multiview import MVDTCWTSPRunner

@click.command()
@click.option('--data-path')
@click.option('--period', default=24*3600)
@click.option('--subperiod', default=3600)
@click.option('--dataset', default='e4')
@click.option('--save-load-dir', default='.')
@click.option('--save', default=False)
@click.option('--load', default=False)
def run_it_all_day_bb(
    data_path, 
    period,
    subperiod,
    dataset,
    save_load_dir,
    save,
    load):

    runner = MVDTCWTSPRunner(
        data_path,
        period=period,
        subperiod=subperiod,
        dataset=dataset,
        save_load_dir=save_load_dir,
        save=save,
        load=load)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
