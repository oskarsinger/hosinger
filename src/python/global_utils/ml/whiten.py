import numpy as np

from sklearn.decomposition import PCA

def get_pca(data, n_comps=0.9):

    pca = PCA(n_components=n_comps)

    pca.fit(data)

    return pca.components_

def get_whitened_data(data):

    (true_n, p) = data.shape
    split_point = p/2
    X = get_pca(data[:n,:split_point])
    Y = get_pca(data[:n,split_point:])
    min_n = min([X.shape[0], Y.shape[0]])
    X = X[:min_n,:]
    Y = Y[:min_n,:]

    return (X, Y)
