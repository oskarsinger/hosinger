from cca import AppGradCCA
from linal.utils import quadratic as quad

import numpy as np

def main():

    n = 10000
    (p1, p2) = (200, 500)
    k = 5
    (eta1, eta2) = (0.1, 0.1)
    (eps1, eps2) = (0.0001, 0.0001)
    reg = 0.0000000001
    batch_size = 100

    X = np.random.randn(n, p1)
    Y = np.random.randn(n, p2)

    app_grad = AppGradCCA(X, Y, k, batch_size=batch_size, eta1=eta1, eta2=eta2)

    (Phi, unn_Phi, Psi, unn_Psi) = app_grad.get_cca()
    Sx = np.dot(X.T, X) / n
    Sy = np.dot(Y.T, Y) / n

    print np.linalg.norm(quad(Phi, A=Sx) - np.identity(k))
    print np.linalg.norm(quad(Psi, A=Sy) - np.identity(k))

if __name__=='__main__':
    main()
