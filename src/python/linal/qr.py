import numpy as np

from linal.utils import get_safe_power

def get_q(A, inner_prod=np.dot):

    (n, p) = A.shape

    x_0 = A[:,0]
    r = (inner_prod(x_0, x_0))**(0.5)
    Q = x_0[:,np.newaxis] / r

    for j in xrange(1,p):
        x_j = A[:,j]
        r_j = get_safe_power(inner_prod(Q, x_j), 0.5)
        y_j = x_j - np.dot(Q, r)
        Q = np.concatenate((Q, y_j), axis=1)

    return np.copy(Q)
