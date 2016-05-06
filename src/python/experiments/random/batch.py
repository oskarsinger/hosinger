from cca.app_grad import AppGradCCA, NViewAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader as GL
from data.servers.gram import BatchGramServer as BGS
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

import numpy as np

def test_batch_appgrad(
    n, p1, p2, cca_k, 
    dl_k1=None, dl_k2=None, 
    optimizer1=MAG(), optimizer2=MAG()):

    if dl_k1 is None:
        dl_k1 = p1

    if dl_k2 is None:
        dl_k2 = p2

    X_loader = GL(n, p1, dl_k1)
    Y_loader = GL(n, p2, dl_k2)
    X_server = BGS(X_loader, 0.01)
    Y_server = BGS(Y_loader, 0.01)
    model = AppGradCCA(
        cca_k)

    model.fit(
        X_server, Y_server,
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        verbose=True)

    return model.get_bases()

def test_batch_n_view_appgrad(
    n, ps, cca_k,
    dl_ks=None,
    optimizers=None):

    if dl_ks is None:
        dl_ks = ps

    if optimizers is None:
        optimizers = [MAG() for i in range(len(ps))]

    loaders = [GL(n, p, dl_k)
               for (p, dl_k) in zip(ps, dl_ks)]
    servers = [BGS(loader, 0.01)
               for loader in loaders]
    model = NViewAppGradCCA(cca_k, len(ps))

    model.fit(
        servers,
        optimizers=optimizers,
        verbose=True)

    return model.get_bases()

def run_two_view_tests(
    n, p1, p2, k):

    print "Parameters:\n\t", "\n\t".join([
        "n: " + str(n),
        "p1: " + str(p1),
        "p2: " + str(p2),
        "k: " + str(k)])

    print "Testing CCA on low-rank data"
    low_rank = test_batch_appgrad(
        n, p1, p2, k, 
        dl_k1=p1/2, dl_k2=p2/2)

    print "Testing CCA on full-rank data"
    full_rank = test_batch_appgrad(
        n, p1, p2, k)

def run_two_view_tests(
    n, ps, k):

    print "Gaussian random data batch AppGrad CCA tests"
    print "Parameters:\n\t", "\n\t".join([
        "n: " + str(n),
        "ps: " + str(ps),
        "k: " + str(k)])

    print "Testing n-view CCA on low-rank data"
    low_rank = test_batch_n_view_appgrad(
        n, ps, k, 
        dl_k1=p1/2, dl_k2=p2/2)

    print "Testing n-view CCA on full-rank data"
    full_rank = test_batch_n_view_appgrad(
        n, ps, k)
