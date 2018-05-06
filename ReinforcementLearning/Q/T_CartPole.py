import gym
import inspect
import numpy as np
import tensorflow as tf
import pandas as pd

env = gym.make("MountainCar-v0")

e = .9
g = .9  #discount
a = .1  #5e10 #learingRate



def takeaction(state):
    global Q
    _state = str(state)
    is_inTable(_state)
    if np.random.uniform() < e:
        sa = Q.loc[_state,:]
        ret = sa.reindex(np.random.permutation(sa.index)).idxmax()
    else:
        ret = np.random.choice(env.action_space.n)
    return ret

def is_inTable(state):
    global Q
    _state= str(state)
    if _state not in Q.index:
        Q = Q.append(pd.Series([0] * env.action_space.n, index=Q.columns, name=_state))

def train(state,action,state_):
    global Q,r,g,a
    _state, _state_, _acton= str(state), str(state_), action
    is_inTable(_state_)
    Q_pred, Q_tar = Q.loc[_state, _acton], r + g * Q.loc[_state_, :].max()

    Q.ix[_state, action] += (a*(Q_pred-Q_tar))

state = env.reset()
env.render()
while True:
    action = takeaction(state)
    state_,r, done,info = env.step(action)
    train(state,action,state_)
    state = state_
    if done:
        break
