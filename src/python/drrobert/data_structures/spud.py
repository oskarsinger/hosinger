class SparsePairwiseUnorderedDict:

    def __init__(self, num_indexes, default=None, no_double=False):

        self.num_indexes = num_indexes
        self.default = default
        self.no_double = no_double

        self.spud = {(i, j) : None if self.default is None else self.default()
                     for i in range(self.num_indexes)
                     for j in range(i, self.num_indexes)}

        if self.no_double:
            self.spud = {k : v for (k, v) in self.spud.items()
                         if not k[0] == k[1]}

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

    def keys(self, no_double=False):

        keys = self.spud.keys()

        if no_double:
            keys = {k for k in keys
                    if not k[0] == k[1]}

        return keys

    def values(self, no_double=False):

        values = self.spud.values()

        if no_double:
            values = [v for (k,v) in self.spud.items()
                      if not k[0] == k[1]]
                      
        return values

    def items(self, no_double=False):
        
        items = self.spud.items()

        if no_double:
            items = [(k,v) for (k,v) in items
                     if not k[0] == k[1]]

        return items
