import numpy as np

import os

from cca import AppGradCCA
from linal.utils import quadratic as quad

def get_data(data_dir, start, end):

    file_names = [file_name
                  for file_name in os.listdir(data_dir)
                  if 'summary' in file_name]
    
    data_points = []

    for file_name in file_names:
        with open(data_dir + file_name) as f:
            f.readline()

            for line in f:
                processed = [float(word)
                             for word in line.split(',')[start:end]]

                data_points.append(processed)

    return np.array(data_points)

def test_cca(data_dir, k, reg, num_points):

    data = get_data(data_dir, 7, -2)
    (n, p) = data.shape
    split_point = p/2
    X = data[:num_points,:split_point]
    Y = data[:num_points,split_point:]
    Sx = np.dot(X.T, X) / X.shape[0]
    Sy = np.dot(Y.T, Y) / Y.shape[0]
    (Phi, unn_Phi, Psi, unn_Psi) = AppGradCCA(X, Y, k, reg=reg).get_cca()

    print np.linalg.norm(quad(Phi, A=Sx) - np.identity(k))
    print np.linalg.norm(quad(Psi, A=Sy) - np.identity(k))
