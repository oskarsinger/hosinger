import h5py
import os

import numpy as np
import data.loaders.shortcuts as dls

from math import ceil
from random import choice

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip
from data.pseudodata import MissingData
from data.servers.minibatch import Minibatch2Minibatch as M2M

from bokeh.palettes import Spectral11
from bokeh.models.layouts import Column
from bokeh.plotting import output_file

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
    print 'Getting data map'
    data_maps = _get_data_maps(hdf5_path, subject)
    names = [m.keys()[0] for m in data_maps]
    titles = [n + ' vs. Time (days) for subject ' + subject
              for n in names]

    print 'Creating plot'
    # Create plot
    plots = [plot_lines(dm, 'Times (days)', 'Value', t)
             for (t, dm) in zip(titles, data_maps)]
    plot = Column(*plots)

    print 'Creating filepath'
    # Preparing filepath
    prefix = subject + '_'
    filename = get_ts(prefix + 
        'value_vs_time_e4_all_views') + '.html'
    filepath = os.path.join(plot_path, filename)

    output_file(filepath, 'value_vs_time_e4_all_views') 

    return plot

def _get_data_maps(hdf5_path, subject):

    print 'Making loaders'
    loaders = dls.get_e4_loaders(hdf5_path, subject, 1, True)
    print 'Making servers'
    servers = [M2M(dl, 1) for dl in loaders]
    data_maps = []
    
    for ds in servers:
        name = ds.get_status()['data_loader'].name()
        print 'Making data map for ' + name
        values = []
        i = 0

        while not ds.finished():

            update = ds.get_data()

            if not isinstance(update, MissingData):
                avg = np.mean(update)

                values.append(avg)
            else:
                values.append(0)

            i += 1

        scaled_indexes = np.arange(len(values)).astype(float) / (24.0 * 3600.0)

        data_map = {}
        data_map[name] = (scaled_indexes, np.array(values))
        
        data_maps.append(data_map)

    return data_maps
