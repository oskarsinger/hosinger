from cca.app_grad import BatchAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import BatchGramServer as BGS

import numpy as np
import time

def test_batch_appgrad(n, p1, p2, k, comid=True, sparse=False):

    X_loader = GL(n, p1)
    Y_loader = GL(n, p2)
    X_server = BGS(X_loader, 0.01)
    Y_server = BGS(Y_loader, 0.01)
    model = BatchAppGradCCA(
        X_server, 
        Y_server, 
        k, 
        comid=comid,
        sparse=sparse)

    return model.get_cca(verbose=True)

def run_tests(n, p1, p2, k):
    #sparse_comid = test_batch_appgrad(n, p1, p2, k, sparse=True)
    non_sparse_comid = test_batch_appgrad(n, p1, p2, k)
    basic = test_batch_appgrad(n, p1, p2, k, comid=False)
