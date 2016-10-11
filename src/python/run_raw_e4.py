import click

from runners.multiview import E4RawDataPlotRunner as E4RDPR

@click.command()
@click.option('--data-path')
@click.option('--period', default=24*3600)
def run_things_all_day_bb(
    data_path,
    period):

    runner = E4RDPR(
        data_path,
        period=period)

    runner.run()

if __name__=='__main__':
    run_things_all_day_bb()
