import gym
import os
import numpy as np
import tensorflow as tf
from .Classes.LinearControll import *


env = gym.make("CartPole-v1")
state = env.reset()

action_space = env.action_space.n
state_len = len(state)


E_start_value = 1.0
E_end_value = 0.1
Lr_Start = 0.1
Lr_Stop = 0.015
Ep_Start = 5.0
Ep_Stop = 10.0
Iterations = 5e6

E = LinearControlSignal(E_start_value,E_end_value,num_iterations=1e6)
LR = LinearControlSignal(Lr_Start,Lr_Stop,num_iterations=Iterations)
Epocs = LinearControlSignal(Ep_Start,Ep_Stop,num_iterations=Iterations)
UpdateIntervall =LinearControlSignal(E_start_value,E_end_value,num_iterations=Iterations)


x = tf.placeholder(shape=[None,state_len],dtype=tf.float32)
y = tf.placeholder(shape=[None,action_space],dtype=tf.float32)



