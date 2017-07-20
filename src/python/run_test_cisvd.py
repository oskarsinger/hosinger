import click

from linal.testers import ColumnIncrementalSVDTester as CISVDT

@click.command()
@click.option('--k', default=1)
@click.option('--n', default=200)
@click.option('--m', default=150)
def run_things_all_day_bb(
    k,
    n,
    m):

    cisvdt = CISVDT(k, n, m)

    cisvdt.run()

if __name__=='__main__':
    run_things_all_day_bb()
