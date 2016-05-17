from cca.app_grad import AppGradCCA as AGCCA
from cca.app_grad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders import readers
from data.servers.batch import BatchServer as BS
from linal.utils import quadratic as quad

import numpy as np

def test_batch_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k)

    model.fit(
        ds1, ds2,
        verbose=True)

    return model.get_bases()

def test_batch_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list))

    model.fit(
        ds_list,
        verbose=True)

    return model.get_bases()

def test_two_fixed_rate_scalar(
    dir_path, file1, file2, cca_k,
    seconds=1,
    reg1=0.1, reg2=0.1,
    reader1=readers.get_scalar, 
    reader2=readers.get_scalar):

    dl1 = FRL(dir_path, file1, seconds, reader1)
    dl2 = FRL(dir_path, file2, seconds, reader2)
    ds1 = BS(dl1)
    ds2 = BS(dl2)
    (Phi, unn_Phi, Psi, unn_Psi) = test_batch_appgrad(
        ds1, ds2, cca_k)
    I_k = np.identity(cca_k)
    gram1 = ds1.get_batch_and_gram()[1]
    gram2 = ds2.get_batch_and_gram()[1]

    print np.linalg.norm(quad(Phi, gram1) - I_k)
    print np.linalg.norm(quad(Psi, gram2) - I_k)

    return (Phi, Psi)

def test_n_fixed_rate_scalar(
    dir_path, cca_k,
    seconds=10):

    mag = readers.get_magnitude
    vec = readers.get_vector
    sca = readers.get_scalar 
    dls = [
        FRL(dir_path, 'ACC.csv', seconds, mag, 32.0),
        IBI(dir_path, 'IBI.csv', seconds, vec),
        FRL(dir_path, 'BVP.csv', seconds, sca, 64.0),
        FRL(dir_path, 'TEMP.csv', seconds, sca, 4.0),
        FRL(dir_path, 'HR.csv', seconds, sca, 1.0),
        FRL(dir_path, 'EDA.csv', seconds, sca, 4.0)]
    dss = [BS(dl) for dl in dls]
    (basis_pairs, Psi) = test_batch_n_view_appgrad(
        dss, cca_k)

    return (basis_pairs, Psi)
