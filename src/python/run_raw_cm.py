import click

from runners.raw import CMRawDataPlotRunner as CMRDPR

@click.command()
@click.option('--data-path')
@click.option('--save-dir')
@click.option('--period', default=8*3600)
@click.option('--hsx', default=True)
@click.option('--lsx', default=True)
@click.option('--w', default=False)
@click.option('--u', default=False)
def run_things_all_day_bb(
    data_path,
    save_dir,
    period,
    hsx,
    lsx,
    w,
    u):

    runner = CMRDPR(
        data_path,
        save_dir,
        period=period,
        hsx=hsx,
        lsx=lsx,
        w=w,
        u=u)

    runner.run()

if __name__=='__main__':
    run_things_all_day_bb()
