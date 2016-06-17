import h5py

from random import choice
from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from bokeh.plotting import show, output_file

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
    data_map = {k : (np.array(xrange(v.shape[0])), v[:])
                for (k, v) in f_session.items()}
    title = ' '.join([
        'Value vs. Sample for session',
        session,
        'of subject',
        subject])

    # Create plot
    p = plot_lines(data_map, 'Sample', 'Value', title)

    # Preparing filepath
    prefix = subject + '_' + session + '_'
    filename = get_ts(prefix + 
        'value_vs_sample_e4_all_views') + '.html'
    filepath = os.path.join(plot_path, filename)

    output_file(file_path, 'value_vs_sample_e4_all_views') 

    show(p)
