import click

from runners.wavelets import MVDTCWTRunner

@click.command()
@click.option('--data-path', default=None)
@click.option('--period', default=24*3600)
@click.option('--subperiod', default=3600)
@click.option('--max-freqs', default=10)
@click.option('--dataset', default='e4')
@click.option('--save-load-dir', default='.')
@click.option('--save', default=False)
@click.option('--load', default=False)
def run_it_all_day_bb(
    data_path, 
    period,
    subperiod,
    max_freqs,
    dataset,
    save_load_dir,
    save,
    load):

    runner = MVDTCWTRunner(
        data_path=data_path,
        period=period,
        subperiod=subperiod,
        max_freqs=max_freqs,
        dataset=dataset,
        save_load_dir=save_load_dir,
        save=save,
        load=load)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
