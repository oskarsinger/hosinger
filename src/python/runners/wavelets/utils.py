import os
import spancca

import numpy as np

from math import log
from drrobert.data_structures import SparsePairwiseUnorderedDict as SPUD
from sklearn.cross_decomposition import CCA

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

def get_cca_vecs(X1, X2, num_nonzero=None):

    (x_weights, y_weights) = [None] * 2

    if num_nonzero is None:
        if np.any(np.iscomplex(X1)):
            X1 = np.absolute(X1)

        if np.any(np.iscomplex(X2)):
            X2 = np.absolute(X2)

        cca = CCA(n_components=1)

        cca.fit(X1, X2)

        x_weights = cca.x_weights_
        y_weights = cca.y_weights_
    else:
        x_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        y_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        A = get_normed_correlation(X1, X2)
        T = X1.shape[0]
        rank = 7
        (x_weights, y_weights) = spancca.cca(
            A,
            rank,
            T,
            x_project,
            y_project,
            verbose=True)

    return (
        x_weights,
        y_weights)

def get_padded_wavelets(Yh, Yl):

    print 'Yh', Yh
    print 'Yl', Yl
    hi_and_lo = Yh + [Yl]
    num_rows = hi_and_lo[0].shape[0]
    basis = np.zeros(
        (num_rows, len(hi_and_lo)),
        dtype=complex)
    basis[:,0] = np.copy(hi_and_lo[0][:,0])

    for (i, y) in enumerate(hi_and_lo[1:]):
        power = i + 1

        for j in xrange(power):
            max_len = basis[j::2**power,power].shape[0]
            basis[j::2**power,power] = np.copy(y[:max_len,0])

    return basis

def get_sampled_wavelets(Yh, Yl):

    hi_and_lo = Yh + [Yl]

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
