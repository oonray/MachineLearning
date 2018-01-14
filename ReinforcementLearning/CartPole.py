import gym
import numpy as np
import tensorflow as tf

LR = 1e-3

env = gym.make("CartPole-v1")
env.reset()

