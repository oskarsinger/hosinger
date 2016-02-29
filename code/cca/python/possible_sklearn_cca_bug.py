# coding: utf-8
from numpy.random import randn
from cca import get_batch_app_grad_decomp as get_cca
n = 10; p1 = 5; p2 = 6; k = 5; eta1 = 0.01; eta2 = 0.01; epsilon1 = 0.001; epsilon2 = 0.001; X = randn(n, p1); Y = randn(n, p2); decomp = get_cca(X, Y, k, eta1, eta2, epsilon1, epsilon2)
(Phi, unn_Phi, Psi, unn_Psi) = decomp
from linal.utils import quadratic
import numpy as np
from sklearn.cross_decomposition import CCA
model = CCA(n_components=5)
model.fit(X,Y)
sk_Phi = model.x_weights_
sk_Psi = model.y_weights_
# These two are not the same.
sk_Phi
Phi
# These two aren't either.
sk_Psi
Psi
# The unnormalized one is also not identical to to the sklearn one.
unn_Psi
# This command results in a vector full of ones, as it should.
np.diag(quadratic(Phi, np.dot(X.T, X) / n)
)
# This one does not. Maybe I should transpose sk_Phi?
np.diag(quadratic(sk_Phi, np.dot(X.T, X) / n)
)

# Tried it with transpose. Didn't work.
