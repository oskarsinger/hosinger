import copy
import sys

sys.path.append("/home/oskar/GitRepos/OskarResearch/code/delayed_rewards")

from bandits import *
from data_servers.action_maps import *
from data_servers import DelayedRewardDataServer as DRDS

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

def run_test(learner, data_server, T):

    for t in range(T):
        action = learner.get_action()
        rewards = data_server.get_rewards(action)
    
        learner.update_rewards(rewards)
    
def main():

    T = 10
    reward_ps = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    num_actions = len(reward_ps)
    delay_consts = [1] * num_actions

    reward_func = get_bernoulli_action_map(reward_ps)
    delay_func = get_const_action_map(delay_consts)
    ucb1_data_server = DRDS(reward_func, delay_func)
    exp3_data_server = DRDS(reward_func, delay_func)
    tsbb_data_server = DRDS(reward_func, delay_func)
    
    ucb1_bold = bold.BOLD(get_UCB1_factory(), num_actions)
    exp3_bold = bold.BOLD(get_Exp3_factory(0.07), num_actions)
    tsbb_bold = bold.BOLD(get_TSBB_factory(1,1), num_actions)

    print "UCB1"
    run_test(ucb1_bold, ucb1_data_server, 1000)
    print "Exp3"
    run_test(exp3_bold, exp3_data_server, 1000)
    print "TSBB"
    run_test(tsbb_bold, tsbb_data_server, 1000)

    print ucb1_bold.get_status()['history'][-10:]
    print exp3_bold.get_status()['history'][-10:]
    print tsbb_bold.get_status()['history'][-10:]

if __name__=='__main__':
    main()
