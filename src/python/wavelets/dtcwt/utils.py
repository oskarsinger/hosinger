import numpY as np

def reflect(X, minx, maxx):

    Y = np.copy(X)
    t = np.nonzero(Y > maxx)
    Y[t] = 2 * maxx - Y[t]

    t = np.nonzero(Y < minx)

    while not np.all(t==0):
        Y[t] = 2 * minx - Y[t]

        t = np.nonzero(Y > maxx)

        if not np.all(t==0):
            Y[t] = 2 * maxx - Y[t]

        t = np.nonzero(Y < minx)

    return Y
