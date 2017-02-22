import h5py
import os

import numpy as np
import drrobert.network as drn

from simple import CosineLoader as CL
from simple import FakePeriodicGaussianLoader as FPGL
from simple import GaussianLoader as GL
from simple import BatchPhysiologicalTimeSeriesLoader as BPTSL
from e4 import FixedRateLoader as FRL
from supervised import LinearRegressionGaussianLoader as LRGL
from readers import from_num as fn
from at import AlTestSpikeLoader as ATSL
from rl import ExposureShiftedGaussianWithBaselineEffectLoader as ESGWBEL
from drrobert.random import rademacher

def get_er_ESGWBEL(
    num_nodes,
    graph_p=0.6,
    rad_p=0.9,
    mus=None,
    sigmas=None,
    baseline_mus=None,
    baseline_sigmas=None):

    network = drn.get_erdos_renyi(
        num_nodes, graph_p, sym=True)
    adj_lists = drn.get_adj_lists(network)
    signs = rademacher(
        p=rad_p, size=num_nodes).tolist()

    if mus is None:
        mus = np.random.uniform(
            size=num_nodes, 
            low=-5, 
            high=5).tolist()

    if sigmas is None:
        sigmas = np.random.uniform(
            size=num_nodes, low=1, high=5)

    if baseline_mus is None:
        baseline_mus = [0] * num_nodes

    if baseline_sigmas is None:
        baseline_sigmas = [0] * num_nodes

    nodes = []

    for i in xrange(num_nodes):
        node = ESGWBEL(
            signs[i],
            mus[i],
            sigmas[i],
            i,
            baseline_mu=baseline_mus[i],
            baseline_sigma=baseline_sigmas[i])

        nodes.append(node)

    for i in xrange(num_nodes):
        neighbors = [nodes[k]
                     for (k, j) in enumerate(adj_lists[i])
                     if j == 1]

        nodes[i].set_neighbors(neighbors)

    return nodes

def get_cm_loaders_all_subjects(filepath):

    cm_pairs = {}
    c_mt = 'Cortisol'
    m_mt = 'Melatonin'
    d_mt = 'DHEAS'
    hertz = 1.0 / (8 * 3600)
    period = 24 * 3600
    num_periods = 8

    with open(filepath) as f:
        f.readline()
        prev_times = {}

        for line in f:
            items = line.strip().split(',')
            (s, t, c, m, d) = items
            t = int(t)
            c = float(c)
            m = float(m)
            d = float(d)

            if t == -72:
                cm_pairs[s] = ([c], [m], [d])
            elif t <= 104:
                if t - prev_times[s] > 8:
                    (c, m) = [np.nan] * 2

                cm_pairs[s][0].append(c)
                cm_pairs[s][1].append(m)
                cm_pairs[s][2].append(d)

            prev_times[s] = t

    loaders = {}

    for (s, (cs, ms, ds)) in cm_pairs.items():
        c_loader = BPTSL(
            np.array(cs)[:,np.newaxis],
            s,
            c_mt,
            hertz,
            period,
            num_periods)
        m_loader = BPTSL(
            np.array(ms)[:,np.newaxis],
            s,
            m_mt,
            hertz,
            period,
            num_periods)
        d_loader = BPTSL(
            np.array(ds)[:,np.newaxis],
            s,
            d_mt,
            hertz,
            period,
            num_periods)
        loaders[s] = [c_loader, m_loader, d_loader]

    return loaders

def get_atr_loaders():

    (TS1, TS2) = ATRG().get_data()
    subject = 'example1'
    hertz = 1.0 / 60
    period = 24 * 3600
    num_periods = 8
    loader1 = BPTSL(
        TS1, 
        subject,
        'TestRampTS1', 
        hertz,
        period,
        num_periods)
    loader2 = BPTSL(
        TS2, 
        subject,
        'TestRampTS2', 
        hertz,
        period,
        num_periods)

    return {subject : [loader1, loader2]}

def get_ats_loaders(data_path, subject=str(1)):

    subject = 'example' + subject

    return [ATSL(os.path.join(data_path, fn))
            for fn in os.listdir(data_path)
            if subject in fn]

def get_ats_loaders_all_subjects(data_path):

    subject1 = 'example' + str(1)
    subject2 = 'example' + str(2)

    return {
        subject1: get_ats_loaders(
            data_path, subject=str(1)),
        subject2: get_ats_loaders(
            data_path, subject=str(2))}

def get_e4_loaders(hdf5_path, subject, online, max_hertz=0.25):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    get_loader = lambda n, r, u: FRL(
        hdf5_path,
        subject,
        n,
        r,
        max_hertz=max_hertz,
        online=online,
        upper=u)

    return [
        get_loader('EDA', fac, 30),
        get_loader('TEMP', fac, 45),
        get_loader('ACC', mag, None),
        get_loader('BVP', fac, None),
        get_loader('HR', fac, None)]

def get_hr_and_acc(hdf5_path, subject, online=False, max_hertz=0.25):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    get_loader = lambda n, r: FRL(
        hdf5_path,
        subject,
        n,
        r,
        max_hertz=max_hertz,
        online=online)

    return [
        get_loader('ACC', mag),
        get_loader('HR', fac)]

def get_e4_loaders_all_subjects(hdf5_path, online=False, max_hertz=0.25):

    subjects = h5py.File(hdf5_path).keys()
    bad = {'HRV15-0' + n for n in ['15', '07', '08']}

    return {s : get_e4_loaders(
                hdf5_path, 
                s, 
                online, 
                max_hertz=max_hertz)
            for s in subjects
            if s not in bad}

def get_hr_and_acc_all_subjects(hdf5_path, online=False, max_hertz=0.25):

    subjects = h5py.File(hdf5_path).keys()
    bad = {'HRV15-0' + n for n in ['15', '07', '08']}

    return {s : get_hr_and_acc(
                    hdf5_path, 
                    s, 
                    online=online, 
                    max_hertz=max_hertz)
            for s in subjects
            if s not in bad}

def get_LRGL(
    n, 
    ps, 
    ws=None, 
    noises=None, 
    noisys=False, 
    bias=False):

    inner_loaders = [GL(n, p) for p in ps]

    if ws is None:
        ws = [None] * len(ps)

    if noises is None:
        noises = [None] * len(ps)

    if not noisys:
        noisys = [False] * len(ps)

    info = zip(
        inner_loaders,
        ws,
        noises,
        noisys)
    
    return [LRGL(il, w=w, noise=ne, noisy=ny, bias=bias)
            for (il, w, ne, ny) in info]

def get_FPGL(n, ps, hertzes):

    return [FPGL(n, p, h) 
            for (p, h) in zip(ps, hertzes)]

def get_cosine_loaders(
    ps,
    periods,
    amplitudes,
    phases,
    indexes,
    period_noise=False,
    phase_noise=False,
    amplitude_noise=False):

    lens = set([
        len(ps),
        len(periods),
        len(amplitudes),
        len(phases),
        len(indexes)])

    if not len(lens) == 1:
        raise ValueError(
            'Args periods, amplitudes, and phases must all have same length.')

    loader_info = zip(
        ps,
        periods,
        amplitudes,
        phases,
        indexes)

    return [_get_CL(p, n, per, a, ph, i,
                period_noise, phase_noise, amplitude_noise)
            for (p, per, a, ph, i) in loader_info]

def _get_CL(
    p,
    max_rounds,
    period,
    amplitude,
    phase,
    index,
    period_noise,
    phase_noise,
    amplitude_noise):

    return CL(
        p,
        max_rounds=max_rounds,
        period=period,
        amplitude=amplitude,
        phase=phase,
        index=index,
        period_noise=period_noise,
        phase_noise=phase_noise,
        amplitude_noise=amplitude_noise)
