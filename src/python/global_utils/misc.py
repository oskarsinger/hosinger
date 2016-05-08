def get_checklist(keys):

    return {k : False for k in keys}

def multi_zip(ls):

    return [tuple([l[i] for l in ls])
            for i in range(len(ls[0]))]

def get_lrange(l):

    return xrange(len(l))

def unzip(l):

    first = [f for (f, s) in l]
    second = [s for (f, s) in l]

    return (first, second)
