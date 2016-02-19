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

    return get_TSBB
    
def get_fake_data(T):

    reward = 1
    delay = 1

    return [(reward, delay)
            for i in range(T)]

def main():

    T = 10

    ucb1_bold = bold.BOLD(get_UCB1_factory(), 2)
    exp3_bold = bold.BOLD(get_Exp3_factory(0.07), 2)
    tsbb_bold = bold.BOLD(get_TSBB_factory(1,1), 2)

    data = get_fake_data(T)

    for i in range(T):
        current_data = data[:-T+i] 

        ucb1_bold.get_action()
        exp3_bold.get_action()
        tsbb_bold.get_action()

        updates = []

        for j in range(len(current_data)):

            reward, delay = current_data[j]

            if delay >= 0:
                data[j] = (reward, delay-1)

                if delay == 0:
                    updates.append({'value':reward, 'id':j})

            ucb1_bold.update_reward(updates)
            exp3_bold.update_reward(updates)
            tsbb_bold.update_reward(updates)

if __name__=='__main__':
    main()
