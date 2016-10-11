import h5py

from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn

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
