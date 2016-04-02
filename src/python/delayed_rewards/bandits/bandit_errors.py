def raise_no_reward_error():

    raise Exception('This learner is still waiting for a reward.')

def raise_no_action_error():

    raise Exception('This learner has not taken an action after receiving' +
            ' its most recent reward.')
