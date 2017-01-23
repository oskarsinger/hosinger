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

def get_cca_vecs(X1, X2, num_nonzero=None):

    (x_weights, y_weights) = [None] * 2

    if num_nonzero is None:
        if np.any(np.iscomplex(X1)):
            X1 = np.absolute(X1)

        if np.any(np.iscomplex(X2)):
            X2 = np.absolute(X2)

        cca = CCA(n_components=1)

        cca.fit(X1, X2)

        x_weights = cca.x_weights_
        y_weights = cca.y_weights_
    else:
        x_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        y_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        A = get_normed_correlation(X1, X2)
        T = X1.shape[0]
        rank = 7
        (x_weights, y_weights) = spancca.cca(
            A,
            rank,
            T,
            x_project,
            y_project,
            verbose=True)

    return (
        x_weights,
        y_weights)

