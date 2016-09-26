from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn

def get_e4_loaders(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'EDA', seconds, fac, online=online),
        FRL(hdf5_path, subject, 'TEMP', seconds, fac, online=online),
        FRL(hdf5_path, subject, 'ACC', seconds, mag, online=online),
        FRL(hdf5_path, subject, 'BVP', seconds, fac, online=online),
        FRL(hdf5_path, subject, 'HR', seconds, fac, online=online)]

def get_changing_e4_loaders(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'ACC', seconds, mag, online=online),
        #IBI(hdf5_path, subject, 'IBI', seconds, fac, online=online),
        FRL(hdf5_path, subject, 'BVP', seconds, fac, online=online),
        FRL(hdf5_path, subject, 'HR', seconds, fac, online=online)]

def get_hr_and_acc(hdf5_path, subject, seconds, online):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns

    return [
        FRL(hdf5_path, subject, 'ACC', seconds, mag, online=online),
        FRL(hdf5_path, subject, 'HR', seconds, fac, online=online)]
