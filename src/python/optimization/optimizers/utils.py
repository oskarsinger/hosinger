import drrobert.arithmetic as da

def get_avg_search_direction(
    old, 
    new, 
    dual_avg, 
    num_rounds, 
    alpha=1,
    beta=0):

    search_direction = new

    if old is not None:
        if dual_avg:
            search_direction = da.get_running_avg(
                old, new, num_rounds)
        else:
            search_direction = da.get_moving_avg(
                old, new, alpha, beta)

    return search_direction

def get_mirror_update(
    parameters, 
    eta, 
    search_direction, 
    get_dual, 
    get_primal):

    dual_parameters = get_dual(parameters)
    dual_update = dual_parameters - eta * search_direction

    return get_primal(dual_update)
