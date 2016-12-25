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
        unit_name = 'unit'
    else:
        condition = unit_name
        unit = 'name'

    df = _get_dataframe(
        data_map,
        x_name,
        y_name,
        unit_name)
    ax = plt.axes()

    print 'Inside plot_lines, generating tsplot'
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

    names = None
    xs = None
    ys = None
    units = None

    for name, (x, y, u) in data_map.items():

        print 'Creating datamap entry for', name
        print 'Creating name array'
        new_n = np.array(
            [name for i in xrange(x.shape[0])])
        print 'Transposing unit arraw'
        new_n = new_n[:,np.newaxis]
        
        new_u = None

        print 'Creating unit array'
        if u is None:
            new_u = np.ones(x.shape[0])
        else:
            new_u = np.array(
                [u for i in xrange(x.shape[0])])

        print 'Transposing unit arraw'
        new_u = new_u[:,np.newaxis]

        print 'Extending name array'
        names = _extend_vec(names, new_n)
        print 'Extending unit array'
        units = _extend_vec(units, new_u)
        print 'Extending xs array'
        xs = _extend_vec(xs, x)
        print 'Extending ys array'
        ys = _extend_vec(ys, y)

    d = {
        x_name: xs[:,0],
        y_name: ys[:,0],
        'name': names[:,0],
        unit_name: units[:,0]}
    print 'Creating dataframe'
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
