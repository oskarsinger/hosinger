class SparsePairwiseUnorderedDict:

    def __init__(self, num_indexes):

        self.num_indexes = num_indexes
        self.spud = {(i, j) : None
                     for i in xrange(self.num_indexes)
                     for j in xrange(i, self.num_indexes)}

    def insert(self, i, j, v):

        k1 = (i, j)
        k2 = (j, i)

        if k1 in self.spud:
            self.spud[k1] = v
        else:
            self.spud[k2] = v

    def get(self, i, j, key=False):

        k1 = (i, j)
        k2 = (j, i)
        v = None

        if k1 in self.spud:
            v = self.spud[k1]

            if key:
                v = (k1, v)
        else:
            v = self.spud[k2]

            if key:
                v = (k2, v)

        return v

    def keys(self):

        return self.spud.keys()

    def values(self):

        return self.spud.values()

    def items(self):

        return self.spud.items()
