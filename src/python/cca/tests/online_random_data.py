from cca.app_grad import OnlineAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import ExpOnlineGramServer as EGS
from data.servers.gram import BoxcarOnlineGramServer as BGS

import numpy as np

def test_online_appgrad_with_exp_server(X_weight, Y_weight, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = EGS(X_loader, X_weight)
    Y_server = EGS(Y_loader, Y_weight)
    model = OnlineAppGradCCA(X_server, Y_server, k)
    
    return model.get_cca(verbose=True)

def test_online_appgrad_with_boxcar_server(X_window, Y_window, p1, p2, k):

    X_loader = GL(1, p1)
    Y_loader = GL(1, p2)
    X_server = BGS(X_loader, X_weight)
    Y_server = BGS(Y_loader, Y_weight)
    model = OnlineAppGradCCA(X_server, Y_server, k)
    
    return model.get_cca(verbose=True)

def run_tests(k, p1, p2):

    test_online_appgrad_with_exp_server(0.75, 0.75, p1, p2, k)
    test_online_appgrad_with_boxcar_server(10, 10, p1, p2, k)
