import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_plot_path

def plot_lines(
    data_map,
    x_name,
    y_name,
    title,
    unit_name=None):

    condition = None
    unit = None

    if unit_name is None:
        condition = 'name'
        unit = 'unit'
    else:
        condition = unit_name
        unit = 'name'

    df = _get_dataframe(
        data_map,
        x_name,
        y_name,
        unit_name)
    ax = plt.axes()

    sns.tsplot(
        time=x_name,
        value=y_name,
        condition=condition,
        unit=unit,
        data=df)

    ax.set_title(title)

    return ax

def _get_dataframe(
    data_map, 
    x_name, 
    y_name, 
    unit_name):

    names = []
    xs = None
    ys = None
    units = []

    for name, (x, y, u) in data_map.items():
        names.extend(
            [name for i in xrange(x.shape[0])])
        
        if u is None:
            units.extend([1] * x.shape[0])
        else:
            units.extend(u)

        if xs is None:
            xs = x
        else:
            xs = np.vstack([xs, x])

        if ys is None:
            ys = y
        else:
            ys = np.vstack([ys, y])

    d = {
        x_name: xs[:,0].tolist(),
        y_name: ys[:,0].tolist(),
        'name': names,
        unit_name: units}
    df = pd.DataFrame(data=d)

    return df
