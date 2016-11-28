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
    title):

    df = _get_dataframe(
        data_map,
        x_name,
        y_name)
    ax = plt.axes()

    sns.tsplot(
        time=x_name,
        value=y_name,
        condition='name',
        unit='units',
        data=df)

    ax.set_title(title)

    return ax

def _get_dataframe(data_map, x_name, y_name):

    names = []
    xs = None
    ys = None
    units = []

    for name, (x_data, y_data) in data_map.items():
        names.extend(
            [name for i in xrange(x_data.shape[0])])
        units.extend([1] * x_data.shape[0])

        if xs is None:
            xs = x_data
        else:
            xs = np.vstack([xs, x_data])

        if ys is None:
            ys = y_data
        else:
            ys = np.vstack([ys, y_data])

    d = {
        x_name: xs[:,0].tolist(),
        y_name: ys[:,0].tolist(),
        'name': names,
        'units': units}
    df = pd.DataFrame(data=d)

    return df
