from operator import mul
from functools import reduce

def get_checklist(keys):

    return {k : False for k in keys}

def get_range_len(l):

    return range(len(l))

def get_list_range(n):

    return list(range(n))

def unzip(l):

    ls = [[item[i] for item in l]
          for i in get_range_len(l[0])]

    return tuple(ls)

def get_list_mod(l, n):

    if len(l) < n:
        raise ValueError(
            'Argument l must have at least n elements.')
     
    new_length = (len(l) / n) * n

    return l[:new_length]

def prod(l):

    return reduce(mul, l, 1)

def get_nums_as_strings(l):

    return [str(n) for n in l]

def get_nested_list_transpose(nl):

    return [[l[i] for l in nl]
            for i in range(len(nl[0]))]
