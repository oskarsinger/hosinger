import numpy as np

from scipy.stats import pearsonr

def get_pearson_matrix(X1, X2):

    (n, p1) = X1.shape
    p2 = X2.shape[1]

    # Get means
    mu1 = np.nanmean(X1, axis=0)
    mu2 = np.nanmean(X2, axis=0)

    # Get zero-mean vars
    zm_X1 = X1 - mu1
    zm_X2 = X2 - mu2

    corr = np.zeros((p1, p2))

    # Get numerator
    for i in xrange(p1):
        for j in xrange(p2):
            numerator = np.nanmean(
                zm_X1[:,i] * zm_X2[:,j])
            sd1 = np.nanmean(np.power(zm_X1,2))**(0.5)
            sd2 = np.nanmean(np.power(zm_X2,2))**(0.5)
            corr[i,j] = numerator / (sd1 * sd2)

    return corr

