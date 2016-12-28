import click
import matplotlib

matplotlib.use('Cairo')

import numpy as np
import seaborn as sns

from runners.distributed.fsvrg import BNRGMMBanditFSVRGRunner as BNRGMMBFSVRGR
from lazyprojector import plot_lines

@click.command()
@click.option('--num-nodes', default=100)
@click.option('--max-rounds', default=5)
def run_it_all_day_bb(
    num_nodes,
    max_rounds):

    inv_powers = list(xrange(5,10))
    graph_ps = [0.5, 0.9]
    budgets = [5, 50]
    data_map = {}

    for budget in budgets:
        for graph_p in graph_ps:
            for i in inv_powers:
                h = 10**(-i)
                runner = BNRGMMBFSVRGR(
                    num_nodes,
                    budget,
                    max_rounds=max_rounds,
                    h=h,
                    graph_p=graph_p)

                runner.run()

                signs = [l.sign for l in runner.loaders]
                ps = np.hstack(
                    [n.model.ps 
                     for n in runner.bfsvrg.nodes])
                argmaxes = np.argmax(ps, axis=0).tolist()
                sign_hats = [-1 if agmx == 0 else 1
                             for agmx in argmaxes]
                errors = [1 for (s, s_hat) in zip(signs, sign_hats)
                          if not s == s_hat]
                num_errors = sum(errors)

                print 'ERRORS', num_errors

                objs = np.array(runner.objectives)
                obj_means = np.sum(objs, axis=0)[:,np.newaxis]
                x = np.arange(max_rounds)[:,np.newaxis]
                data_map['h=' + str(h)] = (x,obj_means,None)

            title = 'Network interference ' + \
                'objective value vs communication round ' + \
                'with budget ' + str(budget) + ' and ' + \
                'graph p ' + str(int(graph_p*10)) + 'x10^-1'
            path = '_'.join(title.split()) + '.pdf'

            plot_lines(
                data_map,
                'communication round',
                'objective value',
                title).get_figure().savefig(
                path, format='pdf')
            sns.plt.clf()


if __name__=='__main__':
    run_it_all_day_bb()
