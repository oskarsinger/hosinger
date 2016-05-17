from cca.app_grad import AppGradCCA, NViewAppGradCCA
from data.loaders.random import GaussianLoader as GL
from data.servers.minibatch import Batch2Minibatch as B2M
from global_utils.arithmetic import int_ceil_log as icl
from global_utils.misc import get_lrange

import numpy as np

def test_online_appgrad(
    X_weight, Y_weight, p1, p2, k):

    bs = k + icl(k)
    X_loader = GL(10*p1, p1)
    Y_loader = GL(10*p2, p2)
    X_server = B2M(X_loader, bs)
    Y_server = B2M(Y_loader, bs)
    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad(ps, k):

    bs = k + icl(k)
    loaders = [GL(10*p, p) for p in ps]
    servers = [BGS(loader) for loader in loaders]
    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        verbose=True)

    return model.get_bases()

def run_two_view_tests(p1, p2, k):

    print "Parameters:\n\t", "\n\t".join([
        "batch_size: " + str(batch_size),
        "p1: " + str(p1),
        "p2: " + str(p2),
        "k: " + str(k)])

    print "Testing CCA with boxcar-weighted Gram matrices"
    boxcar = test_online_appgrad(
        p1, p2, k)

def run_n_view_tests(ps, k):

    print "Gaussian random data online AppGrad CCA tests"
    print "Parameters:\n\t", "\n\t".join([
        "ps: " + str(ps),
        "k: " + str(k)])

    print "Testing n-view CCA with boxcar-weighted Gram matrices"
    boxcar = test_online_n_view_appgrad(
        ps, k)
