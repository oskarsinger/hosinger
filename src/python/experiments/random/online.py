from cca.app_grad import AppGradCCA, NViewAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import ExpOnlineGramServer as EOGS
from data.servers.gram import BoxcarOnlineGramServer as BOGS
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

import numpy as np

def test_online_appgrad_with_exp_server(X_weight, Y_weight, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = EOGS(X_loader, X_weight, k+log(k))
    Y_server = EOGS(Y_loader, Y_weight, k+log(k))
    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        optimizer1=MAG(), optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_appgrad_with_boxcar_server(X_window, Y_window, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = BOGS(X_loader, X_weight, k+log(k))
    Y_server = BOGS(Y_loader, Y_weight, k+log(k))
    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        optimizer1=MAG(), optimizer2=MAG(),
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad_with_exp_server(weights, ps, k):

    loaders = [GL(1, p) for p in ps]
    servers = [EOGS(loader, weight, k+log(k))
               for (loader, weight) in zip(loaders, weights)]
    optimizers = [MAG() for i in range(len(servers))]
    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        optimizers=optimizers,
        verbose=True)

    return model.get_bases()

def test_online_n_view_appgrad_with_boxcar_server(windows, ps, k):

    loaders = [GL(1, p) for p in ps]
    servers = [BOGS(loader, window, k+log(k))
               for (loader, window) in zip(loaders, windows)]
    optimizers = [MAG() for i in range(len(servers))]
    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        optimizers=optimizers,
        verbose=True)

    return model.get_bases()

def run_two_view_tests(k, p1, p2):

    test_online_appgrad_with_exp_server(0.75, 0.75, p1, p2, k)
    test_online_appgrad_with_boxcar_server(10, 10, p1, p2, k)

def run_n_view_tests(k, ps):

    test_online_n_view_appgrad_with_exp_server([0.75] * len(ps), ps, k)
    test_online_n_view_appgrad_with_boxcar_server([10] * len(ps), ps, k)
