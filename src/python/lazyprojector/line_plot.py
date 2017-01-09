import matplotlib

matplotlib.use('Cairo')

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
    unit_name=None,
    ax=None):

    condition = None
    unit = None

    if unit_name is None:
        condition = 'name'
        unit = 'unit'
        unit_name = 'unit'
    else:
        condition = unit_name
        unit = 'name'

    df = _get_dataframe(
        data_map,
        x_name,
        y_name,
        unit_name)
    ax = plt.axes() if ax is None else ax

    sns.tsplot(
        time=x_name,
        value=y_name,
        condition=condition,
        unit=unit,
        data=df,
        ax=ax)

    ax.set_title(title)

    return ax

def _get_dataframe(
    data_map, 
    x_name, 
    y_name, 
    unit_name):

    names = None
    xs = None
    ys = None
    units = None

    for name, (x, y, u) in data_map.items():
        new_n = np.array(
            [name for i in xrange(x.shape[0])])
        new_n = new_n[:,np.newaxis]
        
        new_u = None

        if u is None:
            new_u = np.ones(x.shape[0])
        else:
            new_u = np.array(
                [u for i in xrange(x.shape[0])])

        new_u = new_u[:,np.newaxis]

        print 'y.shape[0]', y.shape[0]
        print 'ys.shape[0] before', ys.shape[0]
        names = _extend_vec(names, new_n)
        units = _extend_vec(units, new_u)
        xs = _extend_vec(xs, (x)
        ys = _extend_vec(ys, y)
        print 'ys.shape[0] after', ys.shape[0]
        print 'ys after', ys

    d = {
        x_name: xs[:,0],
        y_name: ys[:,0],
        'name': names[:,0],
        unit_name: units[:,0]}
    df = pd.DataFrame(data=d)

    return df

def _extend_vec(old, new):

    extended = None

    if old is None:
        extended = new
    else:
        extended = np.vstack(
            [old, new])

    return extended
