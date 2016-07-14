import h5py
import os

import numpy as np

from random import choice
from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip
from bokeh.plotting import show, output_file
from data.loaders.readers import from_num as fn
from data.loaders.e4 import IBILoader as IBI, FixedRateLoader as FRL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from data.missing import MissingData

from math import ceil
from bokeh.palettes import Spectral11

def plot_e4_hdf5_subject(
    hdf5_path, 
    subject=None,
    plot_path='.'):

    # Open hdf5 file
    f = h5py.File(hdf5_path)

    # Choose subject and extract subject info
    if subject is None:
        index = choice(range(len(f.keys())))
        subject = f.keys()[index]

    # Prepare plotting input
    data_map = _get_data_map(hdf5_path, subject)
    title = ' '.join([
        'Value vs. Time (days) for subject',
        subject])

    # Create plot
    p = plot_lines(
        data_map, 
        'Times (days)', 
        'Value', 
        title,
        colors=Spectral11[:4]+Spectral11[-4:])

    # Preparing filepath
    prefix = subject + '_' + session + '_'
    filename = get_ts(prefix + 
        'value_vs_time_e4_all_views') + '.html'
    filepath = os.path.join(plot_path, filename)

    output_file(filepath, 'value_vs_time_e4_all_views') 

    show(p)

def _get_data_map(hdf5_path, subject):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns
    loaders = [
        FRL(hdf5_path, subject, 'EDA', 1, fac, online=True),
        FRL(hdf5_path, subject, 'TEMP', 1, fac, online=True),
        FRL(hdf5_path, subject, 'ACC', 1, mag, online=True),
        IBI(hdf5_path, subject, 'IBI', 1, fac, online=True),
        FRL(hdf5_path, subject, 'BVP', 1, fac, online=True),
        FRL(hdf5_path, subject, 'HR', 1, fac, online=True)]
    servers = [M2M(dl, 1) for dl in loaders]
    data_map = {}
    
    for ds in servers:
        dl = ds.get_status()['data_loader']
        values = []
        indexes = []
        i = 0

        while not dl.finished():
            
            update = ds.get_data()

            if type(update) is not MissingData:
                avg = np.mean(update)

                values.append(avg)
                indexes.append(i)

            i += 1

        scaled_indexes = np.array(indexes).astype(float) / (24 * 3600)
        data_map[dl.name()] = (scaled_indexes, np.array(values))

    return data_map
