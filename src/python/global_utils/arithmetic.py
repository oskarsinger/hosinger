def get_running_avg(current, new, i):

    # remember to look up stable running average on John Cook's blog
    return ((current * (i-1)) + new) / i
