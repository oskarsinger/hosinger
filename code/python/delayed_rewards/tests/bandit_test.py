from bandits import bold, exp3, ucb1, thompson
from data.servers import action_maps as am
from data.servers import DelayedRewardServer as DRS

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

def run_DRS_test(learner, reward_func, delay_func, T):

    ds = DRS(reward_func, delay_func)

    for t in range(T):
        action = learner.get_action()
        rewards = ds.get_rewards(action)
    
        learner.update_rewards(rewards)
    
def test_ucb1_exp3_tsbb(T):

    reward_ps = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    num_actions = len(reward_ps)
    delay_consts = [1] * num_actions

    reward_func = am.get_bernoulli_action_map(reward_ps)
    delay_func = am.get_const_action_map(delay_consts)
    ucb1_bold = bold.BOLD(get_UCB1_factory(), num_actions)
    exp3_bold = bold.BOLD(get_Exp3_factory(0.07), num_actions)
    tsbb_bold = bold.BOLD(get_TSBB_factory(1,1), num_actions)

    print "UCB1"
    run_DRS_test(ucb1_bold, reward_func, delay_func, T)
    print "Exp3"
    run_DRS_test(exp3_bold, reward_func, delay_func, T)
    print "TSBB"
    run_DRS_test(tsbb_bold, reward_func, delay_func, T)

    return {
        'ucb1': ucb1_bold, 
        'exp3': exp3_bold, 
        'tsbb': tsbb_bold}
