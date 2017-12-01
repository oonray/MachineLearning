import numpy as np

def linreg(weights,inp,bias):
    return np.matmul(weights,inp) + bias