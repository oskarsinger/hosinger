import numpy as np

import os

def get_data_summaries(data_dir):

    file_names = [file_name
                  for file_name in os.listdir(data_dir)
                  if 'summary' in file_name]
    data = []
    split_line = lambda line: enumerate(line.split(',')) 

    for file_name in file_names:
        with open(data_dir + file_name) as f:
            first = f.readline()
            length = len(list(split_line))

            for line in f:
                processed = [float(e) if i == length-1 else e
                             for i, e in split_line(line)]

                data.append(processed)

    numerical = [datum[:-1] for datum in data]
    labels = [datum[-1] for datum in data]

    return {
        'obs': np.array(numerical),
        'labels': labels}

