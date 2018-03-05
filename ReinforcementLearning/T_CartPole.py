import gym
import threading as t
import numpy as np
import tensorflow as tf
from .Classes.LinearControll import *
#from Classes.EpsilonGreedy import *
#from Classes.ReplayMemory import *
#from Classes.NeuralNetwork import *


env = gym.make("CartPole-v1")

state = env.reset()

state_len = len(state)
print(state_len)

itr = 0

E_start_value = 1.0
E_end_value = 0.1

E = LinearControlSignal(E_start_value,E_end_value,num_iterations=1e6)

M_Size = 20000
S_Memory = np.zeros(shape=[M_Size,state_len])
Q_Memory = np.zeros(shape=[M_Size,env.action_space])
End_Game = np.zeros(shape=M_Size,dtype=np.bool)
Rewards = np.zeros(shape=M_Size,dtype=np.float)









