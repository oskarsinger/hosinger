import os

from datetime import datetime

def list_dirs_only(path):

    return [sub for sub in list_full_paths(path)
            if os.path.isdir(sub)]

def list_files_only(path):

    return [sub for sub in list_full_paths(path)
            if os.path.isfile(sub)]

def list_full_paths(path):

    return [os.path.join(path, sub)
            for sub in os.listdir(path)]

def get_visible_only(paths):

    return [path for path in paths
            if not path[0] == '.']

def get_timestamped(name):

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    return timestamp + '-' + name

def init_dir(dir_name, save, sl_dir):

    dir_path = os.path.join(
        sl_dir,
        dir_name)

    if save and not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    return dir_path

