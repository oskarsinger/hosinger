import os

from global_utils.file_io import get_ts_filename

def get_plot_path(filename):

    # Should figure out how to initialize this path from a config file
    plot_dir = '/home/oskar/GitRepos/OskarResearch/plots/'
    ts_filename = get_ts_filename(filename) + '.html'

    return os.path.join(plot_dir, ts_filename)
