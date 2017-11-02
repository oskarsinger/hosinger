import numpy as np

def get_kernel_matrix(kernel, X):

    N = X.shape[0] 
    K = np.zeros((N, N))

    for n in range(N):

        X_n = X[n,:]

        for m in range(n, N):

            K_nm = kernel(X_n, X[m,:])
            K[n,m] = K_nm
            K[m,n] = K_nm

    return K
