import h5py
import os

from simple import CosineLoader as CL
from simple import FakePeriodicGaussianLoader as FPGL
from simple import GaussianLoader as GL
from supervised import LinearRegressionGaussianLoader as LRGL
from e4 import FixedRateLoader as FRL
from e4 import IBILoader as IBI
from readers import from_num as fn
from at import AlTestLoader as ATL
from rl import ExposureShiftedGaussianWithBaselineEffectLoader as ESGWBEL

def get_er_ESGWBEL():

    print 'Poop'

def get_at_loaders(data_path, subject=str(1)):

    subject = 'example' + subject

    return [ATL(os.path.join(data_path, fn))
            for fn in os.listdir(data_path)
            if subject in fn]

def get_at_loaders_all_subjects(data_path):

    subject1 = 'example' + str(1)
    subject2 = 'example' + str(2)

    return {
        subject1: get_at_loaders(
            data_path, subject=str(1)),
        subject2: get_at_loaders(
            data_path, subject=str(2))}

def get_e4_loaders(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'EDA', fac, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'TEMP', fac, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'ACC',  mag, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'BVP', fac, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'HR', fac, seconds=seconds, online=online)]

def get_changing_e4_loaders(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'ACC', mag, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'BVP', fac, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'HR', fac, seconds=seconds, online=online)]

def get_hr_and_acc(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'ACC', mag, seconds=seconds, online=online),
        FRL(hdf5_path, subject, 'HR', fac, seconds=seconds, online=online)]

def get_e4_loaders_all_subjects(hdf5_path, seconds, online):

    subjects = h5py.File(hdf5_path).keys()
    bad = {'HRV15-0' + n for n in ['15', '07', '08']}

    return {s : get_e4_loaders(hdf5_path, s, seconds, online)
            for s in subjects
            if s not in bad}

def get_hr_and_acc_all_subjects(hdf5_path, seconds, online):

    subjects = h5py.File(hdf5_path).keys()
    bad = {'HRV15-0' + n for n in ['15', '07', '08']}

    return {s : get_hr_and_acc(hdf5_path, s, seconds, online)
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
