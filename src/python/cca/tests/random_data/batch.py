from cca.app_grad import BatchAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import BatchGramServer as BGS

import numpy as np

def test_batch_appgrad(
    n, p1, p2, cca_k, 
    dl_k1=None, dl_k2=None, 
    comid1=None, comid2=None,
    verbose=False):

    X_loader = GL(n, p1, dl_k1)
    Y_loader = GL(n, p2, dl_k2)
    X_server = BGS(X_loader, 0.01)
    Y_server = BGS(Y_loader, 0.01)
    model = BatchAppGradCCA(
        X_server, 
        Y_server, 
        cca_k, 
        comid1=comid1,
        comid2=comid2)

    return model.get_cca(verbose=verbose)

def run_tests(
    n, p1, p2, k, 
    comid1_type, comid2_type,
    skip_low_rank=True):

    print "Parameters:\n\t", "\n\t".join([
        "n: " + str(n),
        "p1: " + str(p1),
        "p2: " + str(p2),
        "k: " + str(k)])

    print "Testing COMID CCA"
    comid = test_batch_appgrad(
        n, p1, p2, k,
        comid1=comid1_type(), comid2=comid2_type(),
        verbose=False)

    if not skip_low_rank:
        print "Testing COMID CCA on low-rank data"
        comid_low_rank = test_batch_appgrad(
            n, p1, p2, k, 
            dl_k1=p1/2, dl_k2=p2/2,
            comid1=comid1_type(), comid2=comid2_type(),
            verbose=False)

    print "Testing basic AppGrad CCA"
    basic = test_batch_appgrad(
        n, p1, p2, k,
        verbose=False)
