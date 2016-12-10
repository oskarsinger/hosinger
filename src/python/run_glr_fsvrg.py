import click

from runners.distributed.fsvrg import GaussianLinearRegressionFSVRGRunner as GLRFSVRGR

@click.command()
@click.option('--num-nodes', default=4)
@click.option('--n', default=3000)
@click.option('--p', default=500)
@click.option('--max-rounds', default=5)
@click.option('--h', default=0.01)
@click.option('--noisy', default=False)
def run_it_all_day_bb(
    num_nodes,
    n,
    p,
    max_rounds,
    h,
    noisy):

    runner = GLRFSVRGR(
        num_nodes,
        n,
        p,
        max_rounds=max_rounds,
        h=h,
        noisy=noisy)

    runner.run()

    print runner.objectives

if __name__=='__main__':
    run_it_all_day_bb()
