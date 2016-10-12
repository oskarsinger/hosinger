import os

from data.loaders.at import AlTestLoader as ATL

def get_at_loaders(data_path, subject=str(1)):

    subject = 'example' + subject

    return [ATL(os.path.join(data_path, fn))
            for fn in os.listdir(data_path)
            if subject in fn]

def get_at_loaders_all_subjects(data_path):

    subject1 = 'example' + str(1)
    subject2 = 'example' + str(2)

    return {
        subject1: get_at_loaders(
            data_path, subject=str(1)),
        subject2: get_at_loaders(
            data_path, subject=str(2))}
