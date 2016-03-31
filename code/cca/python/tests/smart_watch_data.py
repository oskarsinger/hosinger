import numpy as np

from app_grad import AppGradCCA
from linal.utils import quadratic as quad
from data.smart_watch import get_data_summaries

from sklearn.decomposition import PCA

def get_pca(data, n_comps=0.9):

    pca = PCA(n_components=n_comps)

    pca.fit(data)

    return pca.components_

def run_test(data_dir, k, reg, n):

    data = get_data_summaries(data_dir)['obs']
    (true_n, p) = data.shape
    split_point = p/2
    X = get_pca(data[:n,:split_point])
    Y = get_pca(data[:n,split_point:])
    min_n = min([X.shape[0], Y.shape[0]])
    X = X[:min_n,:]
    Y = Y[:min_n,:]
    Sx = np.dot(X.T, X) / X.shape[0]
    Sy = np.dot(Y.T, Y) / Y.shape[0]
    (Phi, unn_Phi, Psi, unn_Psi) = AppGradCCA(X, Y, k, reg=reg).get_cca()

    print np.linalg.norm(quad(Phi, A=Sx) - np.identity(k))
    print np.linalg.norm(quad(Psi, A=Sy) - np.identity(k))
