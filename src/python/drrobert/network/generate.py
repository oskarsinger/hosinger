import numpy as np

from scipy.stats import bernoulli
from .misc import is_fully_connected

def get_random_parameter_graph(N, p, threshold=None, dist=True):

    num_params = N * p
    ws = None
    w_mat = None
    G = None

    if threshold is None:
        w = np.random.randn(p, 1)
        w_mat = np.hstack(
            [np.copy(w) for _ in range(N)])
        G = np.ones((N, N))

        for i in range(N):
            G[i,i] = 0
    else:
        w = np.random.randn(num_params, 1)
        w_mat = w.reshape((p, N))

        if dist:
            G = get_thresholded_distance(
                w_mat, threshold)
        else:
            G = get_thresholded_similarity(
                w_mat, threshold)

    Bs = [np.linalg.norm(w_mat[:,n])
          for n in range(N)]
    Bw = max(Bs)
    diffs = [w_mat[:,n] - w_mat[:,m]
             for n in range(N)
             for m in range(n+1, N)
             if G[n,m] == 1]
    distances = [np.linalg.norm(d) for d in diffs]
    Dw = max(distances)
    ws = [w_mat[:,n] for n in range(N)]

    return (ws, Bw, Dw, G)

def get_complete():

    G = np.ones((self.num_nodes, self.num_nodes))

    for n in range(self.num_nodes):
        G[n,n] = 0

    return G

def get_thresholded_distance(X, threshold):

    num_items = X.shape[1]
    distances = np.zeros((num_items, num_items))

    for i in range(num_items):
        Xi = X[:,i]
        for j in range(i, num_items):

            if i == j:
                distances[i,j] = np.inf
            else:
                Xj = X[:,j]
                dist = np.linalg.norm(Xi - Xj)

                distances[i,j] = dist
                distances[j,i] = dist

    not_inf = np.logical_not(distances == np.inf)
    min_dist = np.min(distances[not_inf])
    inv_distances = min_dist / distances
    graph = np.zeros_like(distances)
    graph[inv_distances > threshold] = 1

    return graph

def get_thresholded_similarity(X, threshold):

    similarity = np.dot(X.T, X)

    for i in range(similarity.shape[0]):
        similarity[i,i] = 0

    similarity /= np.max(similarity)
    graph = np.zeros_like(similarity)
    graph[similarity > threshold] = 1

    return graph

def get_erdos_renyi(num_nodes, p, sym=False):

    graph = np.zeros((num_nodes, num_nodes))

    if sym:
        num_rvs = int(num_nodes * (num_nodes - 1) / 2)
        edges = bernoulli.rvs(p, size=num_rvs)
        begin = 0
        end = num_nodes - 1
        
        for i in range(num_nodes):
            graph[i,i+1:] = np.copy(edges[begin:end])
            begin = end
            end += num_nodes - i - 2

        graph = graph + graph.T
    else:
        edges = bernoulli.rvs(p, size=num_nodes**2)
        graph += edges.reshape((num_nodes, num_nodes))

    for i in range(num_nodes):
        graph[i,i] = 0

    return graph

def get_adj_lists(adj_matrix):

    n = adj_matrix.shape[0]
    adj_lists = [[] for i in range(n)]

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                adj_lists[i].append(j)

    return adj_lists
