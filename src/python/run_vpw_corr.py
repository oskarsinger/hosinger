import click
import os

from exploratory.mvc import ViewPairwiseCorrelation as VPWC
from data.servers.masks import Interp1DMask as I1DM
from data.servers.masks import DTCWTMask as DTCWTM
from data.servers.batch import BatchServer as BS
from drrobert.file_io import get_timestamped as get_ts

import data.loaders.shortcuts as dlsh

@click.command()
@click.option('--save-load-dir')
@click.option('--data-path', default=None)
@click.option('--num-subperiods', default=24)
@click.option('--dataset', default='e4')
@click.option('--interpolate', default=False)
@click.option('--show', default=False)
@click.option('--wavelet-load', default=False)
@click.option('--wavelet-save', default=False)
@click.option('--wavelet-dir', default=None)
def run_it_all_day_bb(
    save_load_dir,
    data_path,
    num_subperiods,
    dataset,
    interpolate,
    show,
    wavelet_load,
    wavelet_save,
    wavelet_dir):

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
               for (s, dls) in loaders.items()}

    if interpolate:
        servers = {s : [I1DM(s) for s in dss]
                   for (s, dss) in servers.items()}
        
    if not dataset == 'cm':

        if wavelet_save:
            wavelet_dir = os.path.join(
                wavelet_dir, get_ts('DTCWT'))

            os.mkdir(wavelet_dir)

        servers = {sub : [DTCWTM(
                            server, 
                            wavelet_dir, 
                            load=wavelet_load,
                            save=wavelet_save)
                        for server in dss]
                   for (sub, dss) in servers.items()}

    vpwc = VPWC(
        servers,
        save_load_dir,
        num_subperiods=num_subperiods,
        show=show)

    vpwc.run()

if __name__=='__main__':
    run_it_all_day_bb()
