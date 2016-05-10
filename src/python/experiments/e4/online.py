from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from cca.app_grad import AppGradCCA as AGCCA
from cca.app_grad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.e4 import line_processors as lps
from data.servers.gram import BoxcarOnlineGramServer as BOGS
from data.servers.gram import ExpOnlineGramServer as EOGS
from global_utils.arithmetic import int_ceil_log as icl
from global_utils.misc import multi_zip

def test_online_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k, online=True)

    model.fit(
        ds1, ds2,
        optimizer1=MAG(),
        optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list), online=True)

    model.fit(
        ds_list,
        optimizers=[MAG() for i in range(len(ds_list)+1)],
        verbose=True)

    return model.get_bases()

def test_two_fixed_rate_scalar(
    dir_path, file1, file2, cca_k,
    seconds=1,
    reg1=0.1, reg2=0.1,
    lps1=lps.get_scalar, lps2=lps.get_scalar):

    dl1 = FRL(dir_path, file1, seconds, lps1)
    dl2 = FRL(dir_path, file2, seconds, lps2)
    ds1 = BGS(dl1, reg1)
    ds2 = BGS(dl2, reg2)

    (Phi, unn_Phi, Psi, unn_Psi) = test_batch_appgrad(
        ds1, ds2, cca_k)

    return (Phi, Psi)

def test_n_fixed_rate_scalar(
    dir_path, cca_k,
    seconds=10,
    weights=None,
    regs=None):

    batch_size = cca_k + icl(cca_k)
    file_info = {
        ('ACC.csv', lps.get_magnitude, FRL),
        ('IBI.csv', lps.get_vector, IBI),
        ('BVP.csv', lps.get_scalar, FRL),
        ('TEMP.csv', lps.get_scalar, FRL),
        ('HR.csv', lps.get_scalar, FRL),
        ('EDA.csv', lps.get_scalar, FRL)}

    if regs is None:
        regs = [0.1] * len(file_info)

    dls = [LT(dir_path, name, seconds, lp)
           for name, lp, LT in file_info]
    if weights is not None:
        dss = [EOGS(dl, batch_size, w, reg) 
               for dl, reg, w in multi_zip(dls, regs, weights)]
    else:
        dss = [BOGS(dl, batch_size) for dl, reg in zip(dls, regs)]

    (basis_pairs, Psi) = test_batch_n_view_appgrad(
        dss, cca_k)

    return (basis_pairs, Psi)
