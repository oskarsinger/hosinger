import os

def list_dirs_only(path):

    return [sub for sub in list_full_paths(path)
            if os.path.isdir(sub)]

def list_files_only(path):

    return [sub for sub in list_full_paths(path)
            if os.path.isfile(sub)]

def list_full_paths(path):

    return [os.path.join(path, sub)
            for sub in os.listdir(path)]