import click

from runners.raw import E4RawDataPlotRunner as E4RDPR

@click.command()
@click.option('--data-path')
@click.option('--save-dir')
@click.option('--period', default=24*3600)
@click.option('--interpolate', default=False)
@click.option('--hsx', default=True)
@click.option('--lsx', default=True)
@click.option('--w', default=False)
@click.option('--u', default=False)
def run_things_all_day_bb(
    data_path,
    save_dir,
    period,
    interpolate,
    hsx,
    lsx,
    w,
    u):

    runner = E4RDPR(
        data_path,
        save_dir,
        period=period,
        interpolate=interpolate,
        hsx=hsx,
        lsx=lsx,
        w=w,
        u=u)

    runner.run()

if __name__=='__main__':
    run_things_all_day_bb()
