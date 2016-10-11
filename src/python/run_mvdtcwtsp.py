import click

from runners.multiview import MVDTCWTSPRunner

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir', default='.')
@click.option('--save', default=False)
@click.option('--load', default=False)
def run_it_all_day_bb(
    data_path, 
    save_load_dir,
    save,
    load):

    runner = MVDTCWTSPRunner(
        data_path,
        save_load_dir=save_load_dir,
        save=save,
        load=load)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
