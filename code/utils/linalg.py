import numpy as np

def multi_dot(As):

    B = np.identity(As[0].shape[0])

    for A in As:
        B = np.dot(B, A)

    return B
