import click

from runners.distributed.aide import GaussianLinearRegressionAIDERunner as GLRAR

@click.command()
@click.option('--num-nodes', default=4)
@click.option('--n', default=1000)
@click.option('--p', default=500)
@click.option('--max-rounds', default=5)
@click.option('--dane-rounds', default=50)
@click.option('--tau', default=0.1)
@click.option('--gamma', default=0.8)
@click.option('--mu', default=100)
@click.option('--noisy', default=False)
def run_it_all_day_bb(
    num_nodes,
    n,
    p,
    max_rounds,
    dane_rounds,
    tau,
    gamma,
    mu,
    noisy):

    runner = GLRAR(
        num_nodes,
        n,
        p,
        max_rounds=max_rounds,
        dane_rounds=dane_rounds,
        tau=tau,
        gamma=gamma,
        mu=mu,
        noisy=noisy)

    runner.run()

    print runner.objectives

if __name__=='__main__':
    run_it_all_day_bb()
