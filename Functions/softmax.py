import numpy as np

def do(x):
    return np.exp(x) / np.exp(sum(x))
