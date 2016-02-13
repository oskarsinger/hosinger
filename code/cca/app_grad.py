from utils.linalg import multi_dot
from np.linalg import norm

def get_batch_app_grad_decomp(X, Y, eta1, eta2, epsilon1, epsilon2):

    (n1, p1) = X.shape
    (n2, p2) = Y.shape

    assert n1 == n2, 'Number of data points not equal.'

    n = n1
    p =  min([p1,p2])

    Sx = np.dot(X.T, X) / n
    Sy = np.dot(Y.T, Y) / n

    # Randomly initialize normalized and unnormalized canonical bases for 
    # timesteps t and t+1
    Phi_t = np.randn(p1, p)
    unn_Phi_t = np.randn(p1, p)
    Psi_t = np.randn(p2, p)
    unn_Psi_t = np.randn(p2, p)

    Phi_t1 = np.randn(p1, p)
    unn_Phi_t1 = np.randn(p1, p)
    Psi_t1 = np.randn(p2, p)
    unn_Psi_t1 = np.randn(p2, p)

    converged = False

    while not converged:

        # Take a gradient with respect to X's unnormalized canonical basis
        unn_Phi_grad = np.dot(X.T, (np.dot(X, unn_Phi_t) - np.dot(Y, Psi_t)) ) / n

        # Take a gradient step on X's unnormalized canonical basis
        unn_Phi_t1 = unn_Phi_t - eta1 * unn_Phi_grad

        # Take an SVD of the quadratic norm on X's canonical basis
        U_unn_Phi, s_unn_Phi, V_unn_Phi = get_svd_r(multi_dot([unn_Phi_t1.T, Sx, unn_Phi_t1]))

        # Invert the SVD and multiply into X's unnormalized canonical basis to normalize it
        Phi_t1 = multi_dot([unn_Phi_t1, U_unn_Phi, np.power(s_unn_Phi, -0.5), V_unn_Phi])

        # Follow identical steps for Y's unnormalized canonical basis
        unn_Psi_grad = np.dot(Y.T, (np.dot(Y, unn_Psi_t) - np.dot(X, Phi_t)) ) / n
        unn_Psi_t1 = unn_Psi_t - eta2 * unn_Psi_grad
        U_unn_Psi, s_unn_Psi, V_unn_Psi = get_svd_r(multi_dot([unn_Psi_t1.T, Sy, unn_Psi_t1]))
        Psi_t1 = multi_dot([unn_Psi_t1, U_unn_Psi, np.power(s_unn_Psi, -0.5), V_unn_Psi])

        # Calculate deviation from previous time step using Frobenius norm
        Phi_diff = norm(unn_Phi_t - unn_Phi_t1)
        Psi_diff = norm(unn_Psi_t - unn_Psi_t1)

        # Check if error is below tolerance threshold
        converged = Phi_diff < epsilon1 and Psi_diff < epsilon2

    return (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1)


