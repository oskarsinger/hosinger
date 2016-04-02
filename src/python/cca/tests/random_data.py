from app_grad import AppGradCCA
from linal.utils import quadratic as quad

import numpy as np

def run_test(k, reg, n, p1, p2):

    X = np.random.randn(n, p1)
    Y = np.random.randn(n, p2)

    app_grad = AppGradCCA(X, Y, k)

    (Phi, unn_Phi, Psi, unn_Psi) = app_grad.get_cca()
    Sx = np.dot(X.T, X) / n
    Sy = np.dot(Y.T, Y) / n

    print np.linalg.norm(quad(Phi, A=Sx) - np.identity(k))
    print np.linalg.norm(quad(Psi, A=Sy) - np.identity(k))

    return (Phi, Psi)
