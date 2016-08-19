import numpy as np

from drrobert.arithmetic import get_running_avg as get_ra

def set_gradient(old_gradient, new_gradient, dual_avg, num_rounds):

    cumulative_gradient = None

    if dual_avg and old_gradient is not None:
        # Get averaged gradient if desired
        cumulative_gradient = get_ra(
            old_gradient, new_gradient, num_rounds)
    else:
        # Otherwise, get current gradient
        cumulative_gradient = np.copy(new_gradient)

    return cumulative_gradient

def get_update(parameters, eta, gradient, get_dual, get_primal):

    dual_parameters = get_dual(parameters)
    dual_update = dual_parameters - eta * gradient

    return get_primal(dual_update)
