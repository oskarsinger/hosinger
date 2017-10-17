import numpy as np

# NOTE: must give either perm or d
class RowPermutationMatrix:

    def __init__(self, d=None, perm=None):

        if perm is None: 
            perm = np.random.permutation(np.arange(d))
        else:
            d = perm.shape[0]

        self.perm = perm
        self.d = d
        self.index = np.argsort(perm)

    def get_transform(self, A):

        return A[self.index,:]

class ColumnPermutationMatrix:

    def __init__(self, d=None, perm=None):

        if perm is None: 
            perm = np.random.permutation(np.arange(d))
        else:
            d = perm.shape[0]

        self.perm = perm
        self.d = d
        self.index = np.argsort(perm)

    def get_transform(self, A):

        return A[:,self.index]
