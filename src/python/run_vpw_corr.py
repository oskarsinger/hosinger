import click

from exploratory.mvc import ViewPairwiseCorrelation as VPWC
from data.servers.masks import Interp1DMask as I1DM
from data.servers.masks import DTCWTMask as DTCWTM
from data.servers.batch import BatchServer as BS

import data.loaders.shortcuts as dlsh

@click.command()
@click.option('--data-path', default=None)
@click.option('--save-load-dir')
@click.option('--wavelet-dir')
@click.option('--dataset', default='e4')
@click.option('--interpolate', default=True)
@click.option('--show', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    wavelet_dir,
    dataset,
    interpolate,
    show):

    loaders = None

    if dataset == 'e4':
        loaders = dlsh.get_e4_loaders_all_subjects(
            data_path, None, False)
    elif dataset == 'cm':
        loaders = dlsh.get_cm_loaders_all_subjects(
            data_path)
    elif dataset == 'ats':
        loaders = dlsh.get_ats_loaders_all_subjects(
            data_path)
    elif dataset == 'atr':
        loaders = dlsh.get_atr_loaders()
    elif dataset == 'gr':
        ps = [1] * 2
        hertzes = [1.0/60] * 2
        n = 60 * 24 * 8
        loaders = {'e' + str(i): dls.get_FPGL(n, ps, hertzes)
                   for i in xrange(2)}

    servers = {s : [BS(dl) for dl in dls]
               for (s, dl) in loaders.items()}

    if interpolate:
        servers = {s : [I1DM(s) for s in dss]
                   for (s, dss) in servers.items()}

    runner = VPWC(
        servers,
        save_load_dir,
        show=show)

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
