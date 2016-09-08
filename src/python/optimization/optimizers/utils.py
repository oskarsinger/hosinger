import drrobert.arithmetic as da
import drrobert.debug as drdb

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

    print 'Computing dual parameters'

    dual_parameters = get_dual(parameters)

    drdb.check_for_nan_or_inf(
        dual_parameters, 
        'optimizers.utils get_mirror_update', 
        'dual_parameters')

    print 'Computing dual descent update'

    dual_update = dual_parameters - eta * search_direction

    drdb.check_for_nan_or_inf(
        dual_update, 
        'optimizers.utils get_mirror_update', 
        'dual_update')

    print 'Computing primal parameters'

    primal_parameters = get_primal(dual_update)

    drdb.check_for_nan_or_inf(
        primal_parameters, 
        'optimizers.utils get_mirror_update', 
        'primal_parameters')

    print 'Returning primal parameters'

    return primal_parameters
