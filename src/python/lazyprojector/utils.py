import os

from drrobert.file_io import get_timestamped as get_ts

def get_plot_path(filename):

    # Should figure out how to initialize this path from a config file
    plot_dir = '/home/oskar/GitRepos/OskarResearch/plots/'
    ts_filename = get_ts(filename) + '.html'

    return os.path.join(plot_dir, ts_filename)
