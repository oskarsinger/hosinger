import numpy as np
import drrobert.arithmetic as da

def get_search_direction(
    old, 
    new, 
    dual_avg, 
    num_rounds, 
    alpha, 
    beta):

    search_direction = new

    if old_gradient is not None:
        if dual_avg:
            search_direction = da.get_running_avg(
                old, new, num_rounds)
        elif:
            search_direction = da.get_moving_avg(
                old, new, alpha, beta)

    return search_direction

def get_update(
    parameters, 
    eta, 
    search_direction, 
    get_dual, 
    get_primal):

    dual_parameters = get_dual(parameters)
    dual_update = dual_parameters - eta * search_direction

    return get_primal(dual_update)
