import os

from subprocess import call

def init_path():

    cwd = os.getcwd()
    dir_items = os.listdir(cwd)

    for dir_item in dir_items:

        full = os.path.join(cwd, dir_item)

        if os.path.isdir(full):
            call([
                "export",
                "PYTHONPATH=\"${PYTHONPATH}:" + full + "\""])
