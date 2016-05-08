from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from cca.app_grad import AppGradCCA as AGCCA
from cca.app_grad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import line_processors as lps
from data.servers.gram import BoxcarOnlineGramServer as BOGS
from data.servers.gram import ExpOnlineGramServer as EOGS

def test_online_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k)

    model.fit(
        ds1, ds2,
        optimizer1=MAG(),
        optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list))

    model.fit(
        ds_list,
        optimizers=[MAG() for in range(len(ds_list)+1)],
        verbose=True)

    return model.get_bases()

def test_two_fixed_rate_scalar(
    dir_path, file1, file2, cca_k,
    seconds=1,
    reg1=0.1, reg2=0.1,
    lps1=lps.get_scalar, lps2=lps.get_scalar):

    print "Stuff"
