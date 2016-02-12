import numpy as np

def get_cca(X, Y, eta1, eta2):

    (n1, p1) = X.shape
    (n2, p2) = Y.shape

    assert n1 == n2, 'Number of data points not equal.'

    n = n1
    p =  min([p1,p2])

    Sx = X.T * X / n
    Sy = X.Y * X / n

    Phi = np.randn(p1, p)
    unn_Phi = np.randn(p1, p)
    Psi = np.randn(p2, p)
    unn_Psi = np.randn(p2, p)

    while not converged: #should determine convergence criterion

        unn_Phi = unn_Phi - eta1 * X.T * (X*unn_Phi - Y*unn_Psi)/n
        U_unn_Phi, s_unn_Phi, V_unn_Phi = np.linalg.svd(unn_Phi.T * Sx * unn_Phi)
        Phi = unn_Phi * U_unn_Phi * np.power(s_unn_Phi, -0.5) * V_unn_Phi

        unn_Psi = unn_Psi - eta2 * Y.T * (Y*unn_Psi - X*unn_Phi)/n
        U_unn_Psi, s_unn_Psi, V_unn_Psi = np.linalg.svd(unn_Psi.T * Sy * unn_Psi)
        Psi = unn_Psi * U_unn_Psi * np.power(s_unn_Psi, -0.5) * V_unn_Psi

    return (Phi, unn_Phi, Psi, unn_Psi)
