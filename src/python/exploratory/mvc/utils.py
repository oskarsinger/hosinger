import numpy as np

def get_matched_dims(m1, m2):

    (n1, p1) = m1.shape
    (n2, p2) = m2.shape

    if n1 < n2:
        num_reps = int(float(n2) / n1)
        repped = np.zeros((n2, p1))
        
        for r in xrange(num_reps):
            max_len = repped[r::num_reps,:].shape[0]
            repped[r::num_reps,:] = np.copy(
                m1[:max_len,:])

        m1 = repped

    elif n2 < n1:
        num_reps = int(float(n1) / n2)
        repped = np.zeros((n1, p2))
        
        for r in xrange(num_reps):
            max_len = repped[r::num_reps,:].shape[0]
            repped[r::num_reps,:] = np.copy(
                m2[:max_len,:])

        m2 = repped

    return (m1, m2)
