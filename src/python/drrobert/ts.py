import numpy as np
import pandas as pd

from datetime import datetime
from time import mktime
from multiprocessing import Pool

def get_resampled_data(servers, rates, mt=False):

    resampled = []

    if mt:
        p = Pool(len(servers))
        processes = []

        for (ds, rate) in zip(servers, rates):
            processes.append(p.apply_async(
                get_resampled_view, (ds, rate)))

        for process in processes:
            resampled.append(process.get())
    else:
        for (ds, rate) in zip(servers, rates):
            resampled.append(
                get_resampled_vew(ds, rate))

    return resampled

def get_resampled_view(server, rate):

    data = server.get_data()
    loader = server.get_status()['data_loader']
    dt = loader.get_status()['start_times'][0]
    freq = 1.0 / rate
    dt_index = pd.DatetimeIndex(
        data=get_dt_index(server.rows(), freq, dt))
    series = pd.Series(data=data[:,0], index=dt_index)

    return series.resample('S').pad().as_matrix()

def get_dt_index(num_rows, factor, dt, offset=None):

    start = mktime(dt.timetuple())

    if offset is not None:
        start += offset

    times = (np.arange(num_rows) * factor + start).tolist()

    return [datetime.fromtimestamp(t)
            for t in times]
