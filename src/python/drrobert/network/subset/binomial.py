import numpy as np

class BinomialSubsetServer:

    def __init__(self, 
        binomial_p, 
        node_weights):

        self.binomial_p = binomial_p
        self.node_weights = node_weights
        
        self.num_nodes = self.node_weights.shape[0]

    def get_subset(self):

        subset_size = np.random.binomial(
            self.num_nodes, self.binomial_p)
        subset = np.random.choice(
            self.num_nodes,
            subset_size,
            p=self.node_weights,
            replace=False)

        return subset.tolist()
