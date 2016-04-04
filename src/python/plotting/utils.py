import os

from global_utils.file_io import get_ts_filename

def get_plot_path(filename):

    plot_dir = '/home/oskar/GitRepos/OskarResearch/plots/'
    ts_filename = get_ts_filename(filename) + '.html'

    return os.path.join(plot_dir, ts_filename)
