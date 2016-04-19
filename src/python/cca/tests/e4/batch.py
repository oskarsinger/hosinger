from optimization.comid import MatrixAdaGradCOMID as MAG
from cca.app_grad import BatchAppGradCCA as BAG
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import line_processors as lps
from data.servers.gram import BatchGramServer as BGS
from linal.utils import quadratic as quad

import numpy as np

def test_batch_appgrad(
    ds1, ds2, cca_k,
    comid1=None, comid2=None,
    verbose=False):

    model = BAG(
        ds1, ds2, cca_k,
        comid1=comid1,
        comid2=comid2)

    return model.get_cca(verbose=verbose)

def test_two_fixed_rate_scalar(
    dir_path, file1, file2, cca_k,
    comid1_type=MAG, comid2_type=MAG,
    seconds=1,
    reg1=0.1, reg2=0.1,
    lps1=lps.get_scalar, lps2=lps.get_scalar):

    dl1 = FRL(dir_path, file1, seconds, lps1)
    dl2 = FRL(dir_path, file2, seconds, lps2)
    ds1 = BGS(dl1, reg1)
    ds2 = BGS(dl2, reg2)
    comid1 = comid1_type() if comid1_type is not None else None
    comid2 = comid2_type() if comid2_type is not None else None

    (Phi, unn_Phi, Psi, unn_Psi) = test_batch_appgrad(
        ds1, ds2, cca_k, 
        comid1=comid1, 
        comid2=comid2)
    I_k = np.identity(cca_k)
    gram1 = ds1.get_batch_and_gram()[1]
    gram2 = ds2.get_batch_and_gram()[1]

    print np.linalg.norm(quad(Phi, gram1) - I_k)
    print np.linalg.norm(quad(Psi, gram2) - I_k)

    return (Phi, Psi)
