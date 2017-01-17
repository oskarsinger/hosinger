import numpy as np

from scipy.stats import pearsonr

def get_pearson_matrix(X1, X2):

    p1 = X1.shape[1]
    p2 = X2.shape[1]
    corr = np.zeros((p1, p2))

    for i in xrange(p1):
        for j in xrange(p2):
            corr[i,j] = pearsonr(
                X1[:,i], X2[:,j])[0]

    return corr

