import numpy as np
import tensorflow as tf
import pandas as pd

memory_size = 500
batch = 32
memory_counter = 0

state_len = 4
actions = 2

memory = np.zeros((memory_size, state_len+2+state_len))


def store_transition(s, a, r, s_):
    index = memory_counter % memory_size
    memory[index,:] = [i for i in s] + [a,r] + [i for i in s_]
