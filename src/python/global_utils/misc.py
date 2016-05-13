def get_checklist(keys):

    return {k : False for k in keys}

def multi_zip(ls):

    return [tuple([l[i] for l in ls])
            for i in range(len(ls[0]))]

def get_lrange(l):

    return xrange(len(l))

def unzip(l):

    ls = [[item[i] for item in l]
          for i in get_lrange(l[0])]

    return tuple(ls)

def get_list_mod(l, n):

    if len(l) < n:
        raise ValueError(
            'Argument l must have at least n elements.')
     
    new_length = (len(l) / n) * n

    return l[:new_length]
