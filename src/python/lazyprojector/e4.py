import h5py
import os

import numpy as np

from random import choice
from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip
from bokeh.plotting import show, output_file
from data.loaders.readers import from_num as fn

from math import ceil
from bokeh.palettes import Spectral11

def plot_e4_hdf5_session(
    hdf5_path, 
    subject=None, session=None, 
    plot_path='.'):

    # Open hdf5 file
    f = h5py.File(hdf5_path)

    # Choose subject and extract subject info
    if subject is None:
        index = choice(range(len(f.keys())))
        subject = f.keys()[index]

    f_subject = f[subject]

    # Choose session and extract session info
    if session is None:
        index = choice(range(len(f_subject.keys())))
        session = f_subject.keys()[index]

    f_session = f_subject[session]

    # Prepare plotting input
    data_map = _get_data_map(f_session)
    title = ' '.join([
        'Value vs. Sample for session',
        session,
        'of subject',
        subject])

    # Create plot
    p = plot_lines(
        data_map, 
        'Sample', 
        'Value', 
        title,
        colors=Spectral11[:4]+Spectral11[-4:])

    # Preparing filepath
    prefix = subject + '_' + session + '_'
    filename = get_ts(prefix + 
        'value_vs_sample_e4_all_views') + '.html'
    filepath = os.path.join(plot_path, filename)

    output_file(filepath, 'value_vs_sample_e4_all_views') 

    show(p)

def _get_data_map(session):

    num_samples = [v.shape[0] 
                   for (k,v) in session.items()
                   if not 'tags' == k]
    width = min(num_samples)
    window_sizes = [ns / width for ns in num_samples]

    data_map = {}

    # Accelerometer data needs special treatment
    acc = [fn.get_magnitude(entry) 
           for entry in session['ACC']]

    # So does inter-beat interval data
    ibi = _get_ibi_values(session['IBI'])

    # All other streams can be treated the same
    data_map = {k : v[k.lower()]
                for (k, v) in session.items()
                if k not in {'ACC', 'IBI', 'tags'}}

    # Reintroduce the two special guys
    data_map['ACC'] = np.array(acc)
    data_map['IBI'] = np.array(ibi)

    # Average over timesteps to get same length
    normed = _get_normed_streams(data_map)

    return {k : (np.arange(v.shape[0]), v)
            for (k, v) in normed.items()}

def _get_normed_streams(data_map):

    lengths = {k : v.shape[0]
               for (k, v) in data_map.items()}

    # New length to fit all streams to
    l = min(lengths.values())

    # New width and cutoff for each stream
    avging_info = {k : (v / l, v % l)
                   for (k, v) in lengths.items()}

    # Execute cutoff
    truncated = {k : data_map[k][:-c] if c > 0 else data_map[k]
                 for (k, (w, c)) in avging_info.items()}

    # Normalize length of data streams
    reshaped = {k : truncated[k].reshape((l,w))
                for (k, (w, c)) in avging_info.items()}

    # Average over rows or normed data streams
    return {k : np.mean(v, axis=1)
            for (k, v) in reshaped.items()}
    
def _get_ibi_values(ibi_data):
    
    # Last second in which an event occurs
    end = int(ceil(ibi_data[-1][0]))

    # Initialize iteration variables
    seconds = []
    i = 1
    
    # Iterate until entire window is after all recorded events
    while i - 1 < end:
        (event_count, ibi_data) = _get_event_count(ibi_data, i)
        seconds.append(event_count)
        i += 1

    return seconds

def _get_event_count(data, i):

    events = []

    for j, (time, value) in enumerate(data):

        # If time of measurement is outside window
        if time >= i:
            # Truncate measurements already seen
            data = data[j:]

            break

        events.append(value)

    # Give statistic of events occuring in these seconds
    event_count = len(events)

    return (event_count, data)