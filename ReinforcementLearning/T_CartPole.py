import gym
import numpy as np
import tensorflow as tf
from .Classes.LinearControll import *
from .Classes.EpsilonGreedy import *
from .Classes.ReplayMemory import *
from .Classes.NeuralNetwork import *

LR = 1e-3

training = True

env = gym.make("CartPole-v1")
env.reset()

action_space = env.action_space.n
action_names = env.unwrapped.get_action_meanings()

epsilon = EpsilonGreedy(action_space)

LerningRate = LinearControlSignal(start_value=1e-3,end_value=1e-5,num_iterations=5e6)
LossLimit =  LinearControlSignal(start_value=0.1,end_value=0.015,num_iterations=5e6)

MaxEpochs = LinearControlSignal(start_value=5.0,end_value=10.0,num_iterations=5e6)

ReplayFraction = LinearControlSignal(start_value=0.1,end_value=1.0,num_iterations=5e6)

ReplayMemory = ReplayMemory(size=200000,num_actions=action_space)

NeuralNetwork = NeuralNetwork(num_actions=action_space,replay_memory=ReplayMemory)

Rewards=[]

def ResetRewards():
    Rewards = []

def pause():
    pass

def resume():
    pass

def stop():
    pass

def run():
    pass

if __name__ == "__main__":
    run()