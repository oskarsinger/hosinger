from bandits import *
from numpy import random

def get_UCB1_factory():

    def get_UCB1(num_actions):

        return ucb1.UCB1(num_actions)

    return get_UCB1

def get_Exp3_factory(gamma):

    def get_Exp3(num_actions):

        return exp3.Exp3(num_actions, gamma)

    return get_Exp3

def get_TSBB_factory(alpha, beta):

    def get_TSBB(num_actions):

        return thompson.BetaBernoulli(num_actions, alpha, beta)

def main():

    ucb1_bold = bold.BOLD(get_UCB1_factory(), 2)
    exp3_bold = bold.BOLD(get_Exp3_factory(0.07), 2)
    tsbb_bold = bold.BOLD(get_TSBB_factory(1,1), 2)

    

if __name__=='__main__':
    main()
