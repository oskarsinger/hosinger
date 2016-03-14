from cca import get_app_grad_decomp as get_cca
from linal.utils import quadratic as quad

import numpy as np

def main():

    n = 1000
    (p1, p2) = (20, 50)
    k = 5
    (eta1, eta2) = (0.1, 0.1)
    (eps1, eps2) = (0.0001, 0.0001)
    reg = 0.001
    batch_size = 10

    X = np.random.randn(n, p1)
    Y = np.random.randn(n, p2) 
    Sx = np.dot(X.T, X) / n
    Sy = np.dot(Y.T, Y) / n

    (Phi, unn_Phi, Psi, unn_Psi) = get_cca(
        X, Y, k, eta1, eta2, eps1, eps2, reg, batch_size)

    print np.linalg.norm(quad(Phi, Sx) - np.identity(k))
    print np.linalg.norm(quad(Psi, Sy) - np.identity(k))

if __name__=='__main__':
    main()
