def magnitude_process_line(line):

    vals = [float(axis.strip())**2
            for axis in line.split(',')]

    return (sum(vals))**(0.5)

def scalar_process_line(line):

    return float(line.strip())
