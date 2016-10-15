import os

import numpy as np

from math import log
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD

def get_cca_vecs(X1, X2):

    cca = CCA(n_components=1)

    cca.fit(X1, X2)

    return np.hstack([
        cca.x_weights_,
        cca.y_weights_])

def get_normed_correlation(X1, X2):

    sd1 = np.std(X1, axis=0)
    sd2 = np.std(X2, axis=0)
    normed_X1 = X1 / sd1
    normed_X2 = X2 / sd2

    return np.dot(normed_X1.T, normed_X2)

def get_sampled_wavelets(Yh, Yl):

    # TODO: figure out why this doesn't work for short time series
    # TODO: figure out what to do with Yl
    hi_and_lo = Yh# + [Yl]

    # Truncate for full-rank down-sampled coefficient matrix
    threshold = log(hi_and_lo[0].shape[0], 2)
    k = 1

    while log(k, 2) + k <= threshold:
        k += 1

    hi_and_lo = hi_and_lo[:k]
    basis = np.zeros(
        (hi_and_lo[-1].shape[0], k),
        dtype=complex)

    for (i, y) in enumerate(hi_and_lo):
        power = k - i - 1
        sample = np.copy(y[::2**power,0])
        basis[:,i] = sample

    return basis

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

def init_dir(dir_name, save, sl_dir):

    dir_path = os.path.join(
        sl_dir,
        dir_name)

    if save and not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    return dir_path

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

def get_2_digit_pair(i, j):

    get_2_digit = lambda x: '0' + str(x) \
        if int(x)/10 == 0 else \
        str(x)

    return '2^' + get_2_digit(i) + ', 2^' + get_2_digit(j)

def get_2_digit(i):

    x_str = str(x)

    if int(x) / 10 == 0:
        x_str = '0' + x_str

    return x_str
    
