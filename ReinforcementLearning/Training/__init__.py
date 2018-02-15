import gym
import numpy as np
import tensorflow as tf
from ..Classes import LinearControlSignal, ReplayMemory
LR = 1e-3

training = True

env = gym.make("CartPole-v1")
env.reset()

action_space = env.action_space.n
action_names = env.unwrapped.get_action_meanings()

epsilon_training=0.05

num_iterations=1e6
start_value=1.0
end_value=0.1
coefficient = (end_value - start_value) / num_iterations
epsilon = epsilon_training

epsilon_linear = LinearControlSignal(start_value,end_value,num_iterations,False)

learning_control = LinearControlSignal(1e-3,1e-5,5e6)

loss_control = LinearControlSignal(0.1,0.015,5e6)

epoc_control = LinearControlSignal(0.1,1.0,5e6)

replay_memory = ReplayMemory(200000,action_space)

def get_epsilon(iteration):
    return epsilon_linear.get_value(iteration)

def get_action(q_values,iteration,training):
    if training:
        epsilon = epsilon_training
    else:
        epsilon = get_epsilon(iteration)

    if np.random.random() < epsilon:
        action = np.random.randint(low=0, high=action_space)
    else:
        action = np.argmax(q_values)

    return action,epsilon

