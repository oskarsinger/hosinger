import os

import numpy as np

from math import log
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from scipy.stats import pearsonr as ssp

def get_symptom_status(subject):

    status = None

    if type(subject) is str:
        if len(subject) > 2:
            subject = subject[-2:]
        
        subject = int(subject)

    # Symptomatic
    Sx = {2, 5, 9, 11, 17, 18, 19, 20, 23}

    # Asymptomatic
    Asx = {7, 8, 12, 21, 22, 24}

    # Wild type
    W = {3, 4, 6, 13}

    if subject in Sx:
        status = 'Sx'
    elif subject in Asx:
        status = 'Asx'
    elif subject in W:
        status = 'W'
    else:
        raise ValueError('Not a valid subject: ' + str(subject))

    return status

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

def get_correlation_storage(
    num_views,
    num_subperiods,
    num_periods,
    subjects):

    print 'Poop'

def get_cca_vecs(X1, X2):

    cca = CCA(n_components=1)

    cca.fit(X1, X2)

    return np.hstack([
        cca.x_weights_,
        cca.y_weights_])

def get_normed_correlation(X1, X2):

    abs_X1 = np.absolute(X1)
    abs_X2 = np.absolute(X2)

    """
    centered_X1 = abs_X1 - np.mean(abs_X1, axis=0)
    centered_X2 = abs_X2 - np.mean(abs_X2, axis=0)
    unnormed = np.dot(centered_X1.T, centered_X2)
    sd1 = np.std(abs_X1, axis=0)
    sd2 = np.std(abs_X2, axis=0)
    sd_op = np.dot(sd1, sd2.T) * (X1.shape[0] - 1)

    return unnormed / sd_op
    """

    p1 = X1.shape[1]
    p2 = X1.shape[1]
    corr = np.zeros((p1, p2))

    for i in xrange(p1):
        for j in xrange(p2):
            corr[i,j] = ssp(
                abs_X1[:,i], abs_X2[:,j])[0]

    return corr

def get_sampled_wavelets(Yh, Yl):

    hi_and_lo = Yh# + [Yl]

    # Truncate for full-rank down-sampled coefficient matrix
    k = None

    for (i, y) in enumerate(hi_and_lo):
        if y.shape[0] > i:
            k = i+1
        else:
            break

    hi_and_lo = hi_and_lo[:k]
    num_rows = hi_and_lo[-1].shape[0]
    basis = np.zeros(
        (num_rows, k),
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
