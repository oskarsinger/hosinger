def get_magnitude(line):

    vals = [float(axis.strip())**2
            for axis in line.split(',')]

    return (sum(vals))**(0.5)

def get_scalar(line):

    return float(line.strip())

def get_vector(line, delimiter=','):

    strings = line.split(delimiter)

    return [float(item) for item in strings]
