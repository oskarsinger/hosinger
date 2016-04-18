from cca.app_grad import BatchAppGradCCA
from linal.utils import quadratic as quad
from data.loaders.random import GaussianLoader
from data.servers.gram import BatchGramServer

import numpy as np

def test_batch_comid_appgrad(n, p1, p2, k):

    X_loader = GaussianLoader(n, p1)
    Y_loader = GaussianLoader(n, p2)
    X_server = BatcGram
