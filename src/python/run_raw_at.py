import click

from runners.raw import ATRawDataPlotRunner as ATRDPR

@click.command()
@click.option('--data-path')
@click.option('--period', default=24*3600)
@click.option('--std', default=False)
def run_things_all_day_bb(
    data_path,
    period,
    std):

    runner = ATRDPR(
        data_path,
        period=period,
        std=std)

    runner.run()

if __name__=='__main__':
    run_things_all_day_bb()
