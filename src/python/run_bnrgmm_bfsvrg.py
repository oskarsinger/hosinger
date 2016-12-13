import click

import numpy as np

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

    signs = [l.sign for l in runner.loaders]
    ps = np.hstack(
        [n.model.ps 
         for n in runner.bfsvrg.nodes])
    argmaxes = np.argmax(ps, axis=0).tolist()
    sign_hats = [-1 if agmx == 0 else 1
                 for agmx in argmaxes]
    errors = sum(
        [1 for (s, s_hat) in zip(signs, sign_hats)
         if s == s_hat])
    print errors

    print runner.objectives

if __name__=='__main__':
    run_it_all_day_bb()
