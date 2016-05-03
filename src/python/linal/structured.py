import numpy as np

from scipy.linalg.import hadamard

def get_normed_hadamard(n):

    return n**(-0.5) * hadamard(n)
