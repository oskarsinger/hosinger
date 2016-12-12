import click

from runners.distributed.fsvrg import BNRGMMBanditFSVRGRunner as BNRGMMBFSVRGR

@click.command()
@click.option('--num-nodes', default=100)
@click.option('--budget', default=5)
@click.option('--max-rounds', default=5)
@click.option('--h', default=0.01)
def run_it_all_day_bb(
    num_nodes,
    budget,
    max_rounds,
    h):

    runner = BNRGMMBFSVRGR(
        num_nodes,
        budget,
        max_rounds=max_rounds,
        h=h)

    runner.run()

    print runner.objectives

if __name__=='__main__':
    run_it_all_day_bb()
