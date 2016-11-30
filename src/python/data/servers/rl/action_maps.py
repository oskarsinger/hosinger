from numpy.random import binomial, geometric

def get_const_func(const):

    def func():

        return const

    return func

def get_bernoulli_func(p, r=1):

    def func():

        return r*binomial(n=1, p=p)

    return func

def get_geometric_func(p):

    def func():

        return geometric(p=p)

    return func

def get_bernoulli_action_map(ps, r=1):

    funcs = [get_bernoulli_func(p,r)
             for p in ps]

    def map(action):

        func = funcs[action]

        return func()

    return map

def get_const_action_map(consts):

    funcs = [get_const_func(const)
             for const in consts]

    def map(action):

        func = funcs[action]

        return func()

    return map

def get_geometric_action_map(ps):

    funcs = [get_geometric_func(p)
             for p in ps]

    def map(action):

        func = funcs[action]
        
        return func()

    return map
