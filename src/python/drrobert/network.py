import numpy as np

from scipy.stats import bernoulli

def get_erdos_renyi(num_nodes, p, sym=False):

    graph = np.zeros((num_nodes, num_nodes))

    if sym:
        num_rvs = num_nodes * (num_nodes - 1) / 2
        edges = bernoulli.rvs(p, size=num_rvs)
        begin = 0
        end = num_nodes - 1
        
        for i in xrange(num_nodes):
            graph[i,i+1:] = np.copy(edges[begin:end])
            begin = end
            end += num_nodes - i - 2

        graph = graph + graph.T
    else:
        edges = bernoulli.rvs(p, size=num_nodes**2)
        graph += edges.reshape((num_nodes, num_nodes))

    return graph

def get_adj_lists(adj_matrix):

    n = adj_matrix.shape[0]
    adj_lists = [[] for i in xrange(n)]

    for i in xrange(n):
        for j in xrange(n):
            if adj_matrix[i, j] == 1:
                adj_lists[i].append(j)

    return adj_lists
