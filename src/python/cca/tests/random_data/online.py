from cca.app_grad import AppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import ExpOnlineGramServer as EOGS
from data.servers.gram import BoxcarOnlineGramServer as BOGS

import numpy as np

def test_online_appgrad_with_exp_server(X_weight, Y_weight, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = EOGS(X_loader, X_weight, k+log(k))
    Y_server = EOGS(Y_loader, Y_weight, k+log(k))
    model = AppGradCCA(k, online=True)
    
    model.fit(X_server, Y_server, verbose=True)

    return model.get_bases()

def test_online_appgrad_with_boxcar_server(X_window, Y_window, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = BOGS(X_loader, X_weight, k+log(k))
    Y_server = BOGS(Y_loader, Y_weight, k+log(k))
    model = AppGradCCA(k, online=True)
    
    model.fit(X_server, Y_server, verbose=True)

    return model.get_bases()

def run_tests(k, p1, p2):

    test_online_appgrad_with_exp_server(0.75, 0.75, p1, p2, k)
    test_online_appgrad_with_boxcar_server(10, 10, p1, p2, k)
