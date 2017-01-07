import click

from runners.wavelets import E4RawDataPlotRunner as E4RDPR

@click.command()
@click.option('--data-path')
@click.option('--period', default=24*3600)
@click.option('--missing', default=False)
@click.option('--complete', default=False)
@click.option('--std', default=False)
@click.option('--avg-over-subjects', default=False)
def run_things_all_day_bb(
    data_path,
    period,
    missing,
    complete,
    std,
    avg_over_subjects):

    runner = E4RDPR(
        data_path,
        period=period,
        missing=missing,
        complete=complete,
        std=std,
        avg_over_subjects=avg_over_subjects)

    runner.run()

if __name__=='__main__':
    run_things_all_day_bb()
