from cca.app_grad import OnlineAppGradCCA, BatchAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader
from data.servers.gram import ExpOnlineGramServer, BoxcarOnlineGramServer

import numpy as np

def test_online_cca_with_exp_server(X_weight, Y_weight, p1, p2, k):

    X_loader = GaussianLoader(1, p1)
    Y_loader = GaussianLoader(1, p2)
    X_server = ExpOnlineGramServer(X_loader, X_weight)
    Y_server = ExpOnlineGramServer(Y_loader, Y_weight)
    model = OnlineAppGradCCA(X_server, Y_server, k)
    
    return model.get_cca(verbose=True)

def test_online_cca_with_boxcar_server(X_window, Y_window, p1, p2, k):

    X_loader = GaussianLoader(1, p1)
    Y_loader = GaussianLoader(1, p2)
    X_server = BoxcarOnlineGramServer(X_loader, X_weight)
    Y_server = BoxcarOnlineGramServer(Y_loader, Y_weight)
    model = OnlineAppGradCCA(X_server, Y_server, k)
    
    return model.get_cca(verbose=True)

def run_tests(k, p1, p2):

    test_online_cca_with_exp_server(0.75, 0.75, p1, p2, k)
    test_online_cca_with_boxcar_server(10, 10, p1, p2, k)
