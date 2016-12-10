import numpy as np

def get_minibatch(A, batch_size):

    indexes = np.random.choice(
        A.shape[0],
        replace=False, 
        size=batch_size)

    return A[indexes,:]

