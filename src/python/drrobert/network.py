from scipy.stats import bernoulli

def get_erdos_renyi(num_nodes, p, sym=False):

    graph = np.zeros((num_nodes, num_nodes))

    if sym:
        num_rvs = num_nodes * (num_nodes - 1) / 2
        edges = bernoulli.rvs(p, size=num_rvs)
        begin = 0
        end = num_nodes
        
        for i in xrange(num_nodes):
            graph[i,i:] = np.copy(edges[begin:end])
            begin += end
            end += num_nodes - i - 1

        graph = graph + graph.T
    else:
        edges = bernoulli.rvs(p, size=num_nodes**2)
        graph += edges.reshape((num_nodes, num_nodes))

    return graph
