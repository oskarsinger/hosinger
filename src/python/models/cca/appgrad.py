import numpy as np

from drrobert.stats import get_zm_uv
from linal.svd import get_svd_power

# TODO: cite AppGrad paper
# TODO: try to make this more efficient in terms of calls to zmuv and other stuff
class AppGradModel:

    def __init__(self, 
        d1, 
        d2, 
        k=1, 
        idn=None):

        self.d1 = d1
        self.d2 = d2
        self.k = k
        self.idn = idn

    def get_prediction(self, data, params):

        return np.dot(get_zm_uv(data), params)

    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)

        return np.linalg.norm(residuals)**2 / 2

    def get_gradient(self, data, params):

        residuals = self.get_residuals(data, params)
        (X1, X2) = data
        N = X1.shape[0]
        zmuvX1 = get_zm_uv(X1)
        zmuvX2 = get_zm_uv(X2)
        grad1 = np.dot(zmuvX1.T, residuals) / N
        grad2 = np.dot(zmuvX2.T, -residuals) / N

        return (grad1, grad2)

    def get_residuals(self, data, params):

        (Phi1, Phi2) = params
        (X1, X2) = data
        zmuvX1 = get_zm_uv(X1)
        zmuvX2 = get_zm_uv(X2)
        XPhi1 = np.dot(zmuvX1, Phi1)
        XPhi2 = np.dot(zmuvX2, Phi2)

        return XPhi1 - XPhi2

    def get_datum(self, data, i):

        (X1, X2) = data

        return (X1[i,:], X2[i,:])

    def get_projected(self, data, params):

        (X1, X2) = data
        (Phi1, Phi2) = params

        # Get empirical covariance matrices
        zmuvX1 = get_zm_uv(X1)
        zmuvX2 = get_zm_uv(X2)

        # Get projected data
        XPhi1 = np.dot(zmuvX1, Phi1)
        XPhi2 = np.dot(zmuvX2, Phi2)

        # Get gram matrices
        S1 = np.dot(XPhi1.T, XPhi1)
        S2 = np.dot(XPhi2.T, XPhi2)

        # Get normalizers
        inv_srqt1 = get_svd_power(S1, -0.5)
        inv_srqt2 = get_svd_power(S2, -0.5)

        # Get normalized
        normed1 = np.dot(Phi1, inv_sqrt1)
        normed2 = np.dot(Phi2, inv_sqrt2)

        return (normed1, normed2)

    def get_parameter_shape(self):

        phi1_shape = (self.d1, self.k)
        phi2_shape = (self.d2, self.k)

        return (phi1_shape, phi2_shape)
