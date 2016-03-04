from numpy.random import binomial

def get_const_func(const):

    def func():

        return const

    return func

def get_bernoulli_func(p, r=1):

    def func():

        return r*binomial(n=1, p=p)

    return func

def get_bernoulli_reward_func(ps, r=1):

    funcs = [get_bernoulli_func(p,r)
             for p in ps]

    def reward(action):

        func = funcs[action]

        return func()

    return reward

def get_const_reward_func(consts):

    funcs = [get_const_func(const)
             for const in consts]

    def reward(action):

        func = funcs[action]

        return func()

    return reward
