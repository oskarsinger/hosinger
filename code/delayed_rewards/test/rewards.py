from numpy.random import binomial

def get_const_reward_func(const):

    def reward():

        return const

    return reward

def get_bernoulli_reward_func(p, r=1):

    def reward():

        return r*binomial(n=1, p=p)

    return reward
