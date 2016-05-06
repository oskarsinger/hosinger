from cca.app_grad import AppGradCCA, NViewAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import ExpOnlineGramServer as EOGS
from data.servers.gram import BoxcarOnlineGramServer as BOGS
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

import numpy as np

from math import log, ceil

def cl(x):

    return int(ceil(log(x)))

def test_online_appgrad_with_exp_server(
    batch_size, X_weight, Y_weight, p1, p2, k):

    X_loader = GL(2*p1, p1, batch_size=1)
    Y_loader = GL(2*p2, p2, batch_size=1)
    X_server = EOGS(X_loader, k+cl(k), X_weight)
    Y_server = EOGS(Y_loader, k+cl(k), Y_weight)
    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        optimizer1=MAG(), optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_appgrad_with_boxcar_server(
    batch_size, X_window, Y_window, p1, p2, k):

    X_loader = GL(2*p1, p1, batch_size=1)
    Y_loader = GL(2*p2, p2, batch_size=1)
    X_server = BOGS(X_loader, k+cl(k), X_weight)
    Y_server = BOGS(Y_loader, k+cl(k), Y_weight)
    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        optimizer1=MAG(), optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad_with_exp_server(
    batch_size, weights, ps, k):

    loaders = [GL(2*p, p, batch_size=1) for p in ps]
    servers = [EOGS(loader, k+cl(k), weight)
               for (loader, weight) in zip(loaders, weights)]
    optimizers = [MAG() for i in range(len(servers)+1)]
    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        optimizers=optimizers,
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad_with_boxcar_server(
    batch_size, windows, ps, k):

    loaders = [GL(2*p, p, batch_size=1) for p in ps]
    servers = [BOGS(loader, k+cl(k), window)
               for (loader, window) in zip(loaders, windows)]
    optimizers = [MAG() for i in range(len(servers))]
    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        optimizers=optimizers,
        verbose=True)

    return model.get_bases()

def run_two_view_tests(batch_size, p1, p2, k):

    print "Parameters:\n\t", "\n\t".join([
        "batch_size: " + str(batch_size),
        "p1: " + str(p1),
        "p2: " + str(p2),
        "k: " + str(k)])

    print "Testing CCA with exp-weighted Gram matrices"
    exp = test_online_appgrad_with_exp_server(
        batch_size, 0.75, 0.75, p1, p2, k)

    print "Testing CCA with boxcar-weighted Gram matrices"
    boxcar = test_online_appgrad_with_boxcar_server(
        batch_size, 10, 10, p1, p2, k)

def run_n_view_tests(batch_size, ps, k):

    print "Gaussian random data online AppGrad CCA tests"
    print "Parameters:\n\t", "\n\t".join([
        "batch_size: " + str(batch_size),
        "ps: " + str(ps),
        "k: " + str(k)])

    print "Testing n-view CCA with exp-weighted Gram matrices"
    exp = test_online_n_view_appgrad_with_exp_server(
        batch_size, [0.75] * len(ps), ps, k)
    
    print "Testing n-view CCA with boxcar-weighted Gram matrices"
    boxcar = test_online_n_view_appgrad_with_boxcar_server(
        batch_size, [10] * len(ps), ps, k)
