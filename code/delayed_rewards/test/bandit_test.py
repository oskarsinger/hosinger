import copy
import sys

sys.path.append("/home/oskar/GitRepos/OskarResearch/code/delayed_rewards")

from bandits import *
from rewards import *
from delays import *
from numpy.random import geometric

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

    return get_TSBB
    
def get_sim_data(T, reward_func, delay_func):

    return [(reward_func(), delay_func())
            for i in xrange(T)]

def run_test(learner, data):

    data = copy.deepcopy(data)
    T = len(data)

    for i in xrange(T):
        current_data = data[:-T+i] 

        learner.get_action()

        updates = []

        for j in xrange(len(current_data)):

            reward, delay = current_data[j]

            if delay >= -1:
                delay = delay-1
                data[j] = (reward, delay)

            if delay == -1:
                updates.append({'value':reward, 'id':j})

        learner.update_reward(updates)

def main():

    T = 10
    p_reward = 0.8
    p_delay = 0.9

    reward_func = get_bernoulli_reward_func(p_reward)
    delay_func = get_geometric_delay_func(p_delay)
    data = get_sim_data(T, reward_func, delay_func)
    
    print data

    ucb1_bold = bold.BOLD(get_UCB1_factory(), 2)
    exp3_bold = bold.BOLD(get_Exp3_factory(0.07), 2)
    tsbb_bold = bold.BOLD(get_TSBB_factory(1,1), 2)

    print "UCB1"
    run_test(ucb1_bold, data)
    print "Exp3"
    run_test(exp3_bold, data)
    print "TSBB"
    run_test(tsbb_bold, data)

    print ucb1_bold.get_status()
    print exp3_bold.get_status()
    print tsbb_bold.get_status()

if __name__=='__main__':
    main()
