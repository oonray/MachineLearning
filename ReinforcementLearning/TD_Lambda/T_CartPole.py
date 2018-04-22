import gym
import inspect
import numpy as np
import tensorflow as tf
import pandas as pd

env = gym.make("MountainCar-v0")

e = .9
g = .9  #discount
a = .1  #5e10 #learingRate

Q = pd.DataFrame(columns=range(env.action_space.n))

def takeaction(state):
    global Q
    is_inTable(state)
    print(Q)
    if np.random.uniform() < e:
        sa = Q.loc[state, :]
        sa = sa.reindex(np.random.permutation(sa.index))
        print(sa)
        ret = sa.idxmax()
    else:
        ret = np.random.choice(env.action_space.n)
    return ret

def is_inTable(state):
    global Q
    if state not in Q.index:
        Q = Q.append(pd.Series([0]*env.action_space.n, index=Q.columns, name=str(state)))


action = takeaction(str(env.reset()))
print(action)
#env.step(action)
