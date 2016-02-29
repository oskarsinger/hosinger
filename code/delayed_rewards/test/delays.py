from numpy.random import geometric

def get_geometric_delay_func(p):

    def delay():

        return geometric(p=p)

    return delay
