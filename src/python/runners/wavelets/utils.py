import os

import numpy as np

from math import log
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD

def get_complete_status(subject):

    status = None

    if type(subject) in {str, unicode}:
        if len(subject) > 2:
            subject = subject[-2:]
        
        subject = int(subject)

    complete = {}

    return subject in complete

def get_wavelet_storage(
    num_views,
    num_subperiods,
    num_periods,
    subjects):

    get_sp = lambda: [[None, None] 
                      for i in xrange(num_views)]
    get_p = lambda: [get_sp() 
                     for i in xrange(num_subperiods)]
    get_s = lambda s: [get_p() 
                       for i in xrange(num_periods[s])]
    wavelets = {s : get_s(s)
                for s in subjects}

    return wavelets

def get_list_spud_dict(
    num_views, 
    subjects, 
    no_double=False):

    get_list_spud = lambda nd: SPUD(
        num_views, default=list, no_double=nd)

    return {s : get_list_spud(no_double)
            for s in subjects}

def get_appended_spud(list_spud, item_spud):

    for ((i, j), v) in item_spud.items():
        list_spud.get(i, j).append(v)

    return list_spud

def get_kmeans_spud_dict(
    data, 
    label_subjects, 
    k, 
    num_views,
    subjects,
    no_double=False):

    label_spud = get_list_spud_dict(
        num_views,
        subjects,
        no_double=no_double)

    for ((i, j), d) in data.items():
        labels = get_kmeans(
            d, k=k).labels_.tolist()
        subject_list = label_subjects.get(i, j)

        for (l, s) in zip(labels, subject_list):
            label_spud[s].get(i, j).append(l)

    return label_spud

def get_ravel_hstack(matrices):

    cols = [np.ravel(m)[:,np.newaxis]
            for m in matrices]

    return np.hstack(cols)

def get_2_digit_pair(i, j, power=True):

    i_str = get_2_digit(i, power=power)
    j_str = get_2_digit(j, power=power)

    return i_str + ',' + j_str

def get_2_digit(i, power=True):

    i_str = str(i)

    if int(i) / 10 == 0:
        i_str = '0' + i_str

    if power:
        i_str = '2^' + i_str

    return i_str
