import numpy as np

import os

from app_grad import AppGradCCA
from linal.utils import quadratic as quad

from sklearn.decomposition import PCA

def get_data_summaries(data_dir, start_column, end_column):

    file_names = [file_name
                  for file_name in os.listdir(data_dir)
                  if 'summary' in file_name]
    
    data_points = []

    for file_name in file_names:
        with open(data_dir + file_name) as f:
            f.readline()

            for line in f:
                processed = [float(word)
                             for word in line.split(',')[start_column:end_column]]

                data_points.append(processed)

    return np.array(data_points)

def get_pca(data, n_comps=0.9):

    pca = PCA(n_components=n_comps)

    pca.fit(data)

    return pca.components_

def run_test(data_dir, k, reg, n):

    data = get_data_summaries(data_dir, 7, -2)
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
