from cca.app_grad import AppGradCCA as AGCCA
from cca.app_grad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders import readers
from data.servers.minibatch import Batch2Minibatch as B2M
from global_utils.arithmetic import int_ceil_log as icl

def test_online_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k, online=True)

    model.fit(
        ds1, ds2,
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list), online=True)

    model.fit(
        ds_list,
        verbose=True)

    return model.get_bases()

def test_two_fixed_rate_scalar(
    dir_path, file1, file2, cca_k,
    seconds=10,
    reader1=readers.get_scalar, 
    reader2=readers.get_scalar):

    bs = cca_k + icl(cca_k)
    dl1 = FRL(dir_path, file1, seconds, reader1)
    dl2 = FRL(dir_path, file2, seconds, reader2)
    ds1 = B2M(dl1, bs, random=False)
    ds2 = B2M(dl2, bs, random=False)

    (Phi, unn_Phi, Psi, unn_Psi) = test_batch_appgrad(
        ds1, ds2, cca_k)

    return (Phi, Psi)

def test_n_fixed_rate_scalar(
    dir_path, cca_k,
    seconds=10,
    weights=None):

    bs = cca_k + icl(cca_k)
    file_info = {
        ('ACC.csv', reader.get_magnitude, FRL),
        ('IBI.csv', reader.get_vector, IBI),
        ('BVP.csv', reader.get_scalar, FRL),
        ('TEMP.csv', reader.get_scalar, FRL),
        ('HR.csv', reader.get_scalar, FRL),
        ('EDA.csv', reader.get_scalar, FRL)}

    dls = [LT(dir_path, name, seconds, reader)
           for name, reader, LT in file_info]
    dss = [B2M(dl, bs, random=False) for dl in dls]
    (basis_pairs, Psi) = test_batch_n_view_appgrad(
        dss, cca_k)

    return (basis_pairs, Psi)
