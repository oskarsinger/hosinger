import matplotlib

matplotlib.use('Cairo')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data.loaders.shortcuts as dlstcts

from data.servers.batch import BatchServer as BS
from lazyprojector import plot_lines
from drrobert.ts import get_dt_index

class CMRawDataPlotRunner:

    def __init__(self,
        filepath,
        period=24*3600,
        avg_over_subjects=False,
        hsx=True,
        lsx=True,
        w=False,
        u=False):

        self.filepath = filepath
        self.period = period
        self.avg_over_subjects = avg_over_subjects
        self.hsx = hsx
        self.lsx = lsx
        self.w = w
        self.u = u

        self.valid_sympts = set()

        if self.hsx:
            self.valid_sympts.add('Hsx')

        if self.lsx:
            self.valid_sympts.add('Lsx')

        if self.w:
            self.valid_sympts.add('W')

        if self.u:
            self.valid_sympts.add('U')

        self.loaders = dlstcts.get_cm_loaders_all_subjects(
            self.filepath)

    def run(self):

        print 'Poop'
