def get_magnitude(line, delimiter=','):

    squares = [axis**2
               for axis in get_vector(line, delimiter)]

    return (sum(squares))**(0.5)

def get_scalar(line):

    return float(line.strip())

def get_vector(line, delimiter=','):

    strings = line.split(delimiter)

    return [float(item) for item in strings]
